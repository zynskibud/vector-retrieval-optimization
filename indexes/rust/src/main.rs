//! `bench`: build one index, run the queries, write one JSON file (CONTRACT sections 2 to 4).

use bench::{npy, AnnIndex, BuildContext, Matrix, ParamValue, Params, SearchResult};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

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
    Ok(args)
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

    let index = build_index(args, vectors, &build_params)?;
    let times = index.build_times();
    let build = BuildReport {
        train_s: times.train_s,
        add_s: times.add_s,
        total_s: times.train_s + times.add_s,
        peak_rss_mb: peak_rss_mb(),
        index_bytes: index.index_bytes(),
    };

    warm_up(index.as_ref(), &queries, args, &search_sets[0])?;
    let searches = search_sets
        .into_iter()
        .map(|p| timed_search(index.as_ref(), &queries, args.k, p))
        .collect::<Result<Vec<_>, _>>()?;

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
        extra: index.extra(),
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

/// Builds inside a rayon pool of `--threads` threads, so every parallel step uses it.
fn build_index(
    args: &Args,
    vectors: Matrix,
    params: &Params,
) -> Result<Box<dyn AnnIndex>, BenchError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .map_err(|e| Runtime(format!("cannot start thread pool: {e}")))?;
    let ctx = BuildContext {
        out_path: args.out.clone(),
    };
    pool.install(|| bench::build(&args.index, vectors, params, args.threads, args.seed, &ctx))
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
