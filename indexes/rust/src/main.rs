//! `bench`: build one index, run the queries, write one JSON file (CONTRACT sections 2 to 4).

use bench::{npy, AnnIndex, BuildContext, Matrix, ParamValue, Params, SearchResult};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// A failure, with the exit code of CONTRACT section 2.
enum BenchError {
    /// Bad command line, unknown index or parameter: exit code 2.
    Usage(String),
    /// Any other failure: exit code 1.
    Runtime(String),
}

use BenchError::{Runtime, Usage};

struct Args {
    index: String,
    data: String,
    out: String,
    k: usize,
    build: Vec<String>,
    search: Vec<String>,
    threads: usize,
    seed: u64,
    warmup: usize,
    limit: Option<usize>,
    /// Load run (CONTRACT 12.1): worker threads, seconds, insert rows per second.
    clients: usize,
    duration: f64,
    insert_rate: f64,
    /// Changes (CONTRACT 13.2): delete set, update set, compaction and its mode.
    delete: Option<String>,
    update: Option<String>,
    compact: bool,
    compact_mode: String,
}

impl Args {
    /// True for a load run; false keeps the one-pass, one-thread behavior.
    fn load_run(&self) -> bool {
        self.clients != 1 || self.duration > 0.0 || self.insert_rate > 0.0
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&argv).and_then(|a| run(&a)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Usage(msg)) => {
            eprintln!("bench: {msg}");
            ExitCode::from(2)
        }
        Err(Runtime(msg)) => {
            eprintln!("bench: {msg}");
            ExitCode::from(1)
        }
    }
}

fn parse_args(argv: &[String]) -> Result<Args, BenchError> {
    let mut args = Args {
        index: String::new(),
        data: String::new(),
        out: String::new(),
        k: 10,
        build: Vec::new(),
        search: Vec::new(),
        threads: cpu_cores(),
        seed: 42,
        warmup: 100,
        limit: None,
        clients: 1,
        duration: 0.0,
        insert_rate: 0.0,
        delete: None,
        update: None,
        compact: false,
        compact_mode: "rebuild".into(),
    };
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .cloned()
                .ok_or_else(|| Usage(format!("{flag} needs a value")))
        };
        match flag.as_str() {
            "--index" => args.index = value()?,
            "--data" => args.data = value()?,
            "--out" => args.out = value()?,
            "--k" => args.k = parse_num(flag, &value()?)?,
            "--build" => args.build.push(value()?),
            "--search" => args.search.push(value()?),
            "--threads" => args.threads = parse_num(flag, &value()?)?,
            "--seed" => args.seed = parse_num(flag, &value()?)?,
            "--warmup" => args.warmup = parse_num(flag, &value()?)?,
            "--limit" => args.limit = Some(parse_num(flag, &value()?)?),
            "--clients" => args.clients = parse_num(flag, &value()?)?,
            "--duration" => args.duration = parse_num(flag, &value()?)?,
            "--insert-rate" => args.insert_rate = parse_num(flag, &value()?)?,
            "--delete" => args.delete = Some(value()?),
            "--update" => args.update = Some(value()?),
            "--compact" => args.compact = true,
            "--compact-mode" => args.compact_mode = value()?,
            other => return Err(Usage(format!("unknown option: {other}"))),
        }
    }
    for (name, v) in [
        ("--index", &args.index),
        ("--data", &args.data),
        ("--out", &args.out),
    ] {
        if v.is_empty() {
            return Err(Usage(format!("{name} is required")));
        }
    }
    if args.threads == 0 {
        return Err(Usage("--threads must be >= 1".into()));
    }
    if args.clients == 0 {
        return Err(Usage("--clients must be >= 1".into()));
    }
    if !(args.duration >= 0.0 && args.duration.is_finite()) {
        return Err(Usage("--duration must be >= 0".into()));
    }
    if !(args.insert_rate >= 0.0 && args.insert_rate.is_finite()) {
        return Err(Usage("--insert-rate must be >= 0".into()));
    }
    if args.load_run() && args.index != "hnsw" {
        return Err(Usage(format!(
            "--clients, --duration and --insert-rate need --index hnsw (CONTRACT 12); {} is read-only",
            args.index
        )));
    }
    check_change_args(&args)?;
    Ok(args)
}

/// Checks `--delete`, `--update`, `--compact`, `--compact-mode` (CONTRACT 13.2).
fn check_change_args(args: &Args) -> Result<(), BenchError> {
    let changes = args.delete.is_some() || args.update.is_some();
    if !changes && !args.compact && args.compact_mode == "rebuild" {
        return Ok(());
    }
    if !bench::CHANGE_INDEXES.contains(&args.index.as_str()) {
        return Err(Usage(format!(
            "--delete, --update and --compact need --index flat, ivf or hnsw (CONTRACT 13), not {}",
            args.index
        )));
    }
    if let Some(d) = &args.delete {
        if !bench::DELETE_NAMES.contains(&d.as_str()) {
            return Err(Usage(format!("unknown delete set: {d} (expected del10, del30 or del50)")));
        }
    }
    if let Some(u) = &args.update {
        if !bench::UPDATE_NAMES.contains(&u.as_str()) {
            return Err(Usage(format!("unknown update set: {u} (expected upd10)")));
        }
    }
    if args.delete.is_some() && args.update.is_some() {
        return Err(Usage("--delete and --update are not combined in one run".into()));
    }
    if args.compact && !changes {
        return Err(Usage("--compact needs --delete or --update".into()));
    }
    match args.compact_mode.as_str() {
        "rebuild" => {}
        "repair" if args.index == "hnsw" => {}
        "repair" => return Err(Usage("--compact-mode repair exists only for hnsw".into())),
        m => return Err(Usage(format!("unknown --compact-mode: {m} (expected rebuild or repair)"))),
    }
    if args.compact_mode != "rebuild" && !args.compact {
        return Err(Usage("--compact-mode needs --compact".into()));
    }
    if args.load_run() {
        return Err(Usage("changes are not combined with a load run".into()));
    }
    Ok(())
}

fn parse_num<T: std::str::FromStr>(flag: &str, text: &str) -> Result<T, BenchError> {
    text.parse()
        .map_err(|_| Usage(format!("{flag}: '{text}' is not a valid number")))
}

/// Applies `KEY=VALUE[,KEY=VALUE...]` specs over the defaults. Unknown keys are usage errors.
fn apply_params(defaults: &Params, specs: &[&str], kind: &str) -> Result<Params, BenchError> {
    let mut params = defaults.clone();
    for pair in specs
        .iter()
        .flat_map(|s| s.split(','))
        .filter(|s| !s.is_empty())
    {
        let (key, val) = pair
            .split_once('=')
            .ok_or_else(|| Usage(format!("{kind} parameter '{pair}' is not KEY=VALUE")))?;
        if !defaults.contains(key) {
            return Err(Usage(format!("unknown {kind} parameter: {key}")));
        }
        if kind == "search" && key == "filter" {
            bench::check_filter_name(val).map_err(Usage)?;
            params.insert(key, ParamValue::Str(val.to_string()));
            continue;
        }
        params.insert(key, ParamValue::parse(val));
    }
    Ok(params)
}

#[derive(Serialize)]
struct Output {
    contract_version: u32,
    language: &'static str,
    index: String,
    data_dir: String,
    n: usize,
    dim: usize,
    q: usize,
    k: usize,
    threads: usize,
    seed: u64,
    build_params: Params,
    build: BuildReport,
    searches: Vec<SearchReport>,
    machine: Machine,
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
struct BuildReport {
    train_s: f64,
    add_s: f64,
    total_s: f64,
    peak_rss_mb: f64,
    index_bytes: u64,
}

#[derive(Serialize)]
struct SearchReport {
    search_params: Params,
    ids: Vec<Vec<i64>>,
    /// `None` (JSON null) stands for a `-inf` padding score.
    scores: Vec<Vec<Option<f64>>>,
    latency_ms: Vec<f64>,
    total_s: f64,
    qps: f64,
    distance_computations: Option<f64>,
    /// Mean per query of each index counter; `{}` if the index reports none.
    extra: BTreeMap<String, f64>,
}

#[derive(Serialize)]
struct Machine {
    os: String,
    arch: String,
    cpu: String,
    cores: usize,
}

fn run(args: &Args) -> Result<(), BenchError> {
    // Validate the index name and every parameter before the slow data load.
    let search_defaults = bench::search_defaults(&args.index)
        .ok_or_else(|| Usage(format!("unknown index: {}", args.index)))?;
    let build_specs: Vec<&str> = args.build.iter().map(String::as_str).collect();
    let probe = bench::build_defaults(&args.index, 0).expect("index name checked");
    apply_params(&probe, &build_specs, "build")?;
    let search_sets = search_sets(&search_defaults, &args.search)?;

    let dir = Path::new(&args.data);
    let vectors = npy::read_f32(&dir.join("vectors.npy"), args.limit).map_err(Runtime)?;
    let queries = npy::read_f32(&dir.join("queries.npy"), None).map_err(Runtime)?;
    if queries.cols != vectors.cols {
        return Err(Runtime(format!(
            "queries have dim {}, vectors have dim {}",
            queries.cols, vectors.cols
        )));
    }
    let (n, dim) = (vectors.rows, vectors.cols);
    let build_defaults = bench::build_defaults(&args.index, n).expect("index name checked");
    let build_params = apply_params(&build_defaults, &build_specs, "build")?;

    // CONTRACT 12.2: with inserts, the build takes the first 90% of the rows.
    let build_rows = (args.insert_rate > 0.0).then(|| (n * 9 / 10).max(1));
    let tail = build_rows.map(|b| vectors.data[b * dim..].to_vec());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .map_err(|e| Runtime(format!("cannot start thread pool: {e}")))?;
    let mut index = build_index(args, &pool, vectors, &build_params, build_rows)?;
    index.set_filter_dir(&args.data);
    let times = index.build_times();
    let build = BuildReport {
        train_s: times.train_s,
        add_s: times.add_s,
        total_s: times.train_s + times.add_s,
        peak_rss_mb: peak_rss_mb(),
        index_bytes: index.index_bytes(),
    };

    let mut top_extra = index.extra();
    let mut search_sets = search_sets;
    apply_changes(args, &pool, index.as_mut(), n, &mut top_extra, &mut search_sets)?;
    warm_up(index.as_ref(), &queries, args, &search_sets[0])?;
    let searches = if args.load_run() {
        let inserter = build_rows.zip(tail).map(|(first, rows)| Inserter {
            first,
            dim,
            rows,
            next: AtomicUsize::new(first),
            batch_ms: std::sync::Mutex::new(Vec::new()),
        });
        let mut runs = search_sets
            .iter()
            .map(|p| load_search(index.as_ref(), &queries, args, p.clone(), inserter.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(ins) = &inserter {
            // Insert tail (CONTRACT 12.2): rows left when the loop ended go in now,
            // untimed and unpaced, before the repair and the after-inserts pass.
            let during_loop = ins.next.load(Ordering::Acquire) - ins.first;
            let tail_start = Instant::now();
            if ins.run(index.as_ref(), f64::INFINITY, &AtomicBool::new(false)) > 0 {
                return Err(Runtime("insert tail failed".into()));
            }
            let insert_tail_s = tail_start.elapsed().as_secs_f64();
            let (a, b) = index.repair().map_err(Runtime)?;
            let mut params = search_sets[0].clone();
            params.insert("phase", ParamValue::Str("after_inserts".into()));
            let mut after = timed_search(index.as_ref(), &queries, args.k, params)?;
            let inserted = ins.next.load(Ordering::Acquire) - ins.first;
            let mut batch_ms = ins.batch_ms.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let p50 = median(&mut batch_ms);
            for (key, v) in [
                ("inserted_rows", inserted as f64),
                ("inserted_during_loop", during_loop as f64),
                ("insert_tail_s", insert_tail_s),
                ("insert_p50_ms", p50),
                ("insert_batches", batch_ms.len() as f64),
                ("build_rows", ins.first as f64),
                ("repair_added_after_inserts", a as f64),
                ("repair_added_unreachable_after_inserts", b as f64),
            ] {
                after.extra.insert(key.into(), v);
                top_extra.insert(key.into(), v.into());
            }
            runs.push(after);
        }
        runs
    } else {
        search_sets
            .into_iter()
            .map(|p| timed_search(index.as_ref(), &queries, args.k, p))
            .collect::<Result<Vec<_>, _>>()?
    };

    let output = Output {
        contract_version: 1,
        language: "rust",
        index: args.index.clone(),
        data_dir: args.data.clone(),
        n,
        dim,
        q: queries.rows,
        k: args.k,
        threads: args.threads,
        seed: args.seed,
        build_params,
        build,
        searches,
        machine: machine_info(),
        extra: top_extra,
    };
    write_json(&args.out, &output)
}

/// One parameter set per `--search`, or the defaults once if none is given.
fn search_sets(defaults: &Params, specs: &[String]) -> Result<Vec<Params>, BenchError> {
    if specs.is_empty() {
        return Ok(vec![defaults.clone()]);
    }
    specs
        .iter()
        .map(|s| apply_params(defaults, &[s.as_str()], "search"))
        .collect()
}

/// Applies `--delete` or `--update`, then `--compact`, each timed (CONTRACT 13.2).
/// Writes the times and sizes to `extra` and tags every search set with
/// `deleted` / `updated` and `compacted`. Without change flags it does nothing.
fn apply_changes(
    args: &Args,
    pool: &rayon::ThreadPool,
    index: &mut dyn AnnIndex,
    n: usize,
    extra: &mut serde_json::Map<String, serde_json::Value>,
    search_sets: &mut [Params],
) -> Result<(), BenchError> {
    if args.delete.is_none() && args.update.is_none() {
        return Ok(());
    }
    let dir = Path::new(&args.data);
    if let Some(name) = &args.delete {
        let mask = npy::read_bool(&dir.join(format!("delete_{name}.npy")), Some(n)).map_err(Runtime)?;
        if mask.len() != n {
            return Err(Runtime(format!("delete_{name}.npy has {} rows, the corpus has {n}", mask.len())));
        }
        let t = Instant::now();
        pool.install(|| index.delete(&mask)).map_err(Runtime)?;
        extra.insert("delete_s".into(), t.elapsed().as_secs_f64().into());
        extra.insert("deleted_rows".into(), mask.iter().filter(|&&d| d).count().into());
    }
    if let Some(name) = &args.update {
        let ids = npy::read_i64_1d(&dir.join(format!("update_{name}_ids.npy"))).map_err(Runtime)?;
        let vecs = npy::read_f32(&dir.join(format!("update_{name}_vectors.npy")), None).map_err(Runtime)?;
        if vecs.rows != ids.len() {
            return Err(Runtime(format!("update_{name}: {} IDs, {} vectors", ids.len(), vecs.rows)));
        }
        // With --limit, only the IDs inside the first n rows apply.
        let keep: Vec<usize> = (0..ids.len()).filter(|&j| (ids[j] as usize) < n).collect();
        let sel_ids: Vec<i64> = keep.iter().map(|&j| ids[j]).collect();
        let sel_vecs: Vec<f32> = keep.iter().flat_map(|&j| vecs.row(j).iter().copied()).collect();
        let t = Instant::now();
        pool.install(|| index.update(&sel_ids, &sel_vecs)).map_err(Runtime)?;
        extra.insert("update_s".into(), t.elapsed().as_secs_f64().into());
        extra.insert("updated_rows".into(), sel_ids.len().into());
    }
    extra.insert("index_bytes_before_compact".into(), index.index_bytes().into());
    if args.compact {
        let t = Instant::now();
        pool.install(|| match args.compact_mode.as_str() {
            "repair" => index.compact_repair(),
            _ => index.compact(),
        })
        .map_err(Runtime)?;
        extra.insert("compact_s".into(), t.elapsed().as_secs_f64().into());
        extra.insert("compact_mode".into(), args.compact_mode.clone().into());
        extra.insert("index_bytes_after".into(), index.index_bytes().into());
    }
    for p in search_sets.iter_mut() {
        if let Some(d) = &args.delete {
            p.insert("deleted", ParamValue::Str(d.clone()));
        }
        if let Some(u) = &args.update {
            p.insert("updated", ParamValue::Str(u.clone()));
        }
        p.insert("compacted", ParamValue::Int(args.compact as i64));
    }
    Ok(())
}

/// Builds inside the rayon pool of `--threads` threads, so every parallel step uses it.
fn build_index(
    args: &Args,
    pool: &rayon::ThreadPool,
    vectors: Matrix,
    params: &Params,
    build_rows: Option<usize>,
) -> Result<Box<dyn AnnIndex>, BenchError> {
    let ctx = BuildContext {
        out_path: args.out.clone(),
    };
    pool.install(|| match build_rows {
        Some(rows) => {
            bench::build_partial(&args.index, vectors, params, args.threads, args.seed, rows)
        }
        None => bench::build(&args.index, vectors, params, args.threads, args.seed, &ctx),
    })
    .map_err(Runtime)
}

/// Runs the first `--warmup` queries once with the first search set, untimed.
fn warm_up(
    index: &dyn AnnIndex,
    queries: &Matrix,
    args: &Args,
    params: &Params,
) -> Result<(), BenchError> {
    for i in 0..args.warmup.min(queries.rows) {
        std::hint::black_box(
            index
                .search(queries.row(i), args.k, params)
                .map_err(Runtime)?,
        );
    }
    Ok(())
}

/// Runs all queries one at a time on this thread. Only the search call is timed per query.
fn timed_search(
    index: &dyn AnnIndex,
    queries: &Matrix,
    k: usize,
    params: Params,
) -> Result<SearchReport, BenchError> {
    let q = queries.rows;
    let mut results: Vec<SearchResult> = Vec::with_capacity(q);
    let mut latency_ms = Vec::with_capacity(q);
    let loop_start = Instant::now();
    for i in 0..q {
        let t = Instant::now();
        let r = index.search(queries.row(i), k, &params);
        latency_ms.push(t.elapsed().as_secs_f64() * 1e3);
        results.push(r.map_err(Runtime)?);
    }
    let total_s = loop_start.elapsed().as_secs_f64();
    Ok(report(params, results, latency_ms, total_s))
}

/// The rows that the inserter adds during load runs (CONTRACT 12.2).
struct Inserter {
    /// First row that is not in the build.
    first: usize,
    dim: usize,
    /// Rows `first..N`, row-major.
    rows: Vec<f32>,
    /// Next row to insert. Shared by all search settings.
    next: AtomicUsize,
    /// Wall time of each batch of 100 rows, in ms.
    batch_ms: std::sync::Mutex<Vec<f64>>,
}

const INSERT_BATCH: usize = 100;

impl Inserter {
    /// Inserts batches of 100 rows at `rate` rows per second until the rows run out
    /// or `stop` is set. Returns the number of failed batches.
    fn run(&self, index: &dyn AnnIndex, rate: f64, stop: &AtomicBool) -> u64 {
        let total = self.first + self.rows.len() / self.dim;
        let start = Instant::now();
        let mut failed = 0u64;
        let mut batch = 0u64;
        while !stop.load(Ordering::Acquire) {
            let lo = self.next.load(Ordering::Acquire);
            if lo >= total {
                break;
            }
            // Batch j starts at j * 100 / rate seconds after the start (rate = inf: at once).
            let due = start + Duration::from_secs_f64(batch as f64 * INSERT_BATCH as f64 / rate);
            let now = Instant::now();
            if now < due {
                std::thread::sleep((due - now).min(Duration::from_millis(5)));
                continue;
            }
            let hi = (lo + INSERT_BATCH).min(total);
            let ids: Vec<i64> = (lo as i64..hi as i64).collect();
            let vecs = &self.rows[(lo - self.first) * self.dim..(hi - self.first) * self.dim];
            let t = Instant::now();
            match index.insert(&ids, vecs) {
                Ok(()) => {
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    self.batch_ms
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(ms);
                    self.next.store(hi, Ordering::Release);
                }
                Err(e) => {
                    eprintln!("bench: insert failed: {e}");
                    failed += 1;
                    break;
                }
            }
            batch += 1;
        }
        failed
    }
}

/// What one load worker returns.
struct WorkerOut {
    latency_ms: Vec<f64>,
    /// Worker 0 only: its first pass over the queries, in query order.
    first_pass: Vec<SearchResult>,
    errors: u64,
}

/// Load run of CONTRACT 12.1 for one search setting: `--clients` threads, each in a
/// closed loop over the queries for `--duration` seconds (0 = one pass each), and the
/// inserter thread if `--insert-rate` > 0. Worker 0 starts at query 0; worker w starts
/// at query w * Q / C, so the workers do not run the same query at the same time.
fn load_search(
    index: &dyn AnnIndex,
    queries: &Matrix,
    args: &Args,
    params: Params,
    inserter: Option<&Inserter>,
) -> Result<SearchReport, BenchError> {
    if !index.supports_concurrency() {
        return Err(Usage("this index does not support load runs".into()));
    }
    let q = queries.rows;
    let (clients, k) = (args.clients, args.k);
    let stop = AtomicBool::new(false);
    let cpu_before = cpu_seconds();
    let loop_start = Instant::now();
    let (outs, insert_failed, wall) = std::thread::scope(|sc| {
        let ins = inserter.map(|ins| sc.spawn(|| ins.run(index, args.insert_rate, &stop)));
        let workers: Vec<_> = (0..clients)
            .map(|w| {
                let (stop, params) = (&stop, &params);
                sc.spawn(move || {
                    let mut out = WorkerOut {
                        latency_ms: Vec::with_capacity(q),
                        first_pass: Vec::with_capacity(if w == 0 { q } else { 0 }),
                        errors: 0,
                    };
                    let offset = w * q / clients;
                    let mut i = 0usize;
                    loop {
                        if args.duration > 0.0 && stop.load(Ordering::Relaxed) {
                            break;
                        }
                        if args.duration == 0.0 && i == q {
                            break;
                        }
                        let t = Instant::now();
                        let r = index.search(queries.row((offset + i) % q), k, params);
                        out.latency_ms.push(t.elapsed().as_secs_f64() * 1e3);
                        let r = r.unwrap_or_else(|e| {
                            if out.errors == 0 {
                                eprintln!("bench: query failed: {e}");
                            }
                            out.errors += 1;
                            SearchResult {
                                ids: vec![-1; k],
                                scores: vec![f32::NEG_INFINITY; k],
                                distance_computations: None,
                                counters: BTreeMap::new(),
                            }
                        });
                        if w == 0 && i < q {
                            out.first_pass.push(r);
                        }
                        i += 1;
                    }
                    out
                })
            })
            .collect();
        if args.duration > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(args.duration));
            stop.store(true, Ordering::Release);
        }
        let outs: Vec<WorkerOut> = workers
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect();
        let wall = loop_start.elapsed().as_secs_f64();
        stop.store(true, Ordering::Release);
        let insert_failed = ins.map_or(0, |h| h.join().expect("inserter panicked"));
        (outs, insert_failed, wall)
    });
    let cpu_pct = (cpu_seconds() - cpu_before) / wall * 100.0;
    let errors: u64 = outs.iter().map(|o| o.errors).sum();
    let done: usize = outs.iter().map(|o| o.latency_ms.len()).sum();
    let mut outs = outs.into_iter();
    let mut first = outs.next().expect("clients >= 1");
    let mut latency_ms = std::mem::take(&mut first.latency_ms);
    outs.for_each(|o| latency_ms.extend(o.latency_ms));
    // A short --duration can end before worker 0 finishes its first pass: complete
    // it after the loop, untimed, so ids and scores always hold Q rows.
    let short_pass = q - first.first_pass.len();
    for i in first.first_pass.len()..q {
        first
            .first_pass
            .push(index.search(queries.row(i), k, &params).map_err(Runtime)?);
    }
    let mut rep = report(params, first.first_pass, latency_ms, wall);
    rep.qps = done as f64 / wall;
    for (key, v) in [
        ("errors", errors as f64),
        ("cpu_pct", cpu_pct),
        ("clients", clients as f64),
        ("duration_s", args.duration),
        ("queries_done", done as f64),
        ("first_pass_completed_after_loop", short_pass as f64),
    ] {
        rep.extra.insert(key.into(), v);
    }
    if inserter.is_some() {
        rep.extra.insert("insert_errors".into(), insert_failed as f64);
        rep.extra.insert("insert_rate".into(), args.insert_rate);
    }
    Ok(rep)
}

/// Median of `v` (sorts it); 0 for an empty list.
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// User plus system CPU time of this process, in seconds.
fn cpu_seconds() -> f64 {
    // SAFETY: as in `peak_rss_mb`: getrusage writes only into the zeroed plain struct.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage
    };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

fn report(
    search_params: Params,
    results: Vec<SearchResult>,
    latency_ms: Vec<f64>,
    total_s: f64,
) -> SearchReport {
    let q = results.len();
    let counts: Option<Vec<u64>> = results.iter().map(|r| r.distance_computations).collect();
    let distance_computations = counts.map(|c| c.iter().sum::<u64>() as f64 / q.max(1) as f64);
    let extra = mean_counters(&results);
    let scores = results
        .iter()
        .map(|r| {
            r.scores
                .iter()
                .map(|&s| s.is_finite().then_some(s as f64))
                .collect()
        })
        .collect();
    SearchReport {
        search_params,
        ids: results.into_iter().map(|r| r.ids).collect(),
        scores,
        latency_ms,
        total_s,
        qps: q as f64 / total_s,
        distance_computations,
        extra,
    }
}

/// Mean per query of every counter key. A query without a key counts as 0 for it.
fn mean_counters(results: &[SearchResult]) -> BTreeMap<String, f64> {
    let mut sums: BTreeMap<String, f64> = BTreeMap::new();
    for (key, v) in results.iter().flat_map(|r| &r.counters) {
        *sums.entry(key.clone()).or_default() += v;
    }
    let q = results.len().max(1) as f64;
    sums.values_mut().for_each(|v| *v /= q);
    sums
}

fn write_json(path: &str, output: &Output) -> Result<(), BenchError> {
    let file = std::fs::File::create(path).map_err(|e| Runtime(format!("{path}: {e}")))?;
    serde_json::to_writer(std::io::BufWriter::new(file), output)
        .map_err(|e| Runtime(format!("{path}: {e}")))
}

fn cpu_cores() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Peak resident set size of this process, in MB (2^20 bytes).
fn peak_rss_mb() -> f64 {
    // SAFETY: `getrusage` only writes into the zeroed struct we pass, which is a
    // plain C struct with no invalid bit patterns, and RUSAGE_SELF is always valid.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage
    };
    let maxrss = usage.ru_maxrss as f64;
    // ru_maxrss is in bytes on macOS and in KB on Linux.
    let bytes = if cfg!(target_os = "macos") {
        maxrss
    } else {
        maxrss * 1024.0
    };
    bytes / (1u64 << 20) as f64
}

fn machine_info() -> Machine {
    // The contract uses the uname names: darwin, arm64.
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "aarch64" if cfg!(target_os = "macos") => "arm64",
        other => other,
    };
    Machine {
        os: os.to_string(),
        arch: arch.to_string(),
        cpu: cpu_name(),
        cores: cpu_cores(),
    }
}

#[cfg(target_os = "macos")]
fn cpu_name() -> String {
    let name = c"machdep.cpu.brand_string";
    let mut buf = [0u8; 256];
    let mut len: libc::size_t = buf.len();
    // SAFETY: `name` is NUL-terminated, `buf` is writable for `len` bytes, and
    // sysctlbyname writes at most `len` bytes and stores the written length in `len`.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return "unknown".into();
    }
    let text = &buf[..len.min(buf.len())];
    let end = text.iter().position(|&b| b == 0).unwrap_or(text.len());
    String::from_utf8_lossy(&text[..end]).trim().to_string()
}

#[cfg(not(target_os = "macos"))]
fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}
