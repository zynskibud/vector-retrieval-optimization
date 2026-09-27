//! The `.vro` index file of CONTRACT 15.1, shared by all four languages.
//!
//! Layout, little-endian:
//! - magic, 8 bytes: `VROIDX01`
//! - header_len, uint32
//! - header, header_len bytes of ASCII JSON: index, n, dim, build_params, seed,
//!   contract_version, language, and the section table (name, dtype, shape, offset, bytes)
//! - zero padding to the next multiple of 64 bytes
//! - the sections, raw arrays in table order, each at an offset that is a multiple of 64
//!
//! The writer computes the offsets from the header length, and the header length
//! depends on the digits of the offsets, so it repeats until the two agree.
//! The reader reads the section table and never assumes an offset.

use crate::{ParamValue, Params, Tombstones};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"VROIDX01";
/// Section offsets are multiples of this.
pub const ALIGN: u64 = 64;
/// Upper bound for header_len, so a corrupt length cannot allocate gigabytes.
const MAX_HEADER: u32 = 1 << 24;
/// Bytes converted per read or write.
const CHUNK_BYTES: usize = 1 << 22;

fn align(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

/// The data of one section, borrowed from the index.
pub enum Data<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
    U8(&'a [u8]),
}

impl Data<'_> {
    fn dtype(&self) -> &'static str {
        match self {
            Data::F32(_) => "f32",
            Data::I32(_) => "int32",
            Data::U8(_) => "u8",
        }
    }
    fn len(&self) -> usize {
        match self {
            Data::F32(d) => d.len(),
            Data::I32(d) => d.len(),
            Data::U8(d) => d.len(),
        }
    }
}

fn dtype_size(dtype: &str) -> Option<u64> {
    match dtype {
        "f32" | "int32" => Some(4),
        "u8" => Some(1),
        _ => None,
    }
}

/// One section to write: its name, shape, and data (row-major, `prod(shape)` values).
pub struct Section<'a> {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub data: Data<'a>,
}

/// One row of the section table.
#[derive(Debug, Clone, PartialEq)]
pub struct SectionInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub bytes: u64,
}

/// The parsed header.
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub index: String,
    pub n: usize,
    pub dim: usize,
    pub build_params: Params,
    pub seed: u64,
    pub contract_version: u64,
    pub language: String,
    pub sections: Vec<SectionInfo>,
    /// header_len of the file.
    pub header_len: u32,
}

impl Header {
    /// The section table row named `name`.
    pub fn section(&self, name: &str) -> Result<&SectionInfo, String> {
        self.sections
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| format!("vro: no section '{name}'"))
    }
}

/// Params as a JSON object: ints, floats, and strings.
pub fn params_to_json(p: &Params) -> Value {
    serde_json::to_value(p).expect("params serialize")
}

/// Params from a JSON object. A number without a fraction becomes an int.
pub fn params_from_json(v: &Value) -> Result<Params, String> {
    let obj = v.as_object().ok_or("vro: build_params is not an object")?;
    let mut p = Params::new();
    for (k, v) in obj {
        let pv = if let Some(i) = v.as_i64() {
            ParamValue::Int(i)
        } else if let Some(f) = v.as_f64() {
            ParamValue::Float(f)
        } else if let Some(s) = v.as_str() {
            ParamValue::Str(s.to_string())
        } else {
            return Err(format!("vro: build_params.{k} is not a number or string"));
        };
        p.insert(k, pv);
    }
    Ok(p)
}

/// True when two parameter values are equal. Numbers compare by value (4 == 4.0).
pub fn param_eq(a: &ParamValue, b: &ParamValue) -> bool {
    use ParamValue::*;
    match (a, b) {
        (Int(x), Int(y)) => x == y,
        (Int(x), Float(y)) | (Float(y), Int(x)) => *x as f64 == *y,
        (Float(x), Float(y)) => x == y,
        (Str(x), Str(y)) => x == y,
        _ => false,
    }
}

/// Writes a `.vro` file. Returns the file size in bytes. The file is synced to disk.
pub fn write(
    path: &Path,
    index: &str,
    n: usize,
    dim: usize,
    build_params: &Params,
    seed: u64,
    sections: &[Section],
) -> Result<u64, String> {
    for s in sections {
        let want: usize = s.shape.iter().product();
        if s.data.len() != want {
            return Err(format!(
                "vro: section {} has {} values, shape {:?} needs {want}",
                s.name,
                s.data.len(),
                s.shape
            ));
        }
    }
    let table = |base: u64| -> (Vec<Value>, u64) {
        let mut off = base;
        let mut rows = Vec::new();
        for s in sections {
            off = align(off);
            let bytes = s.data.len() as u64 * dtype_size(s.data.dtype()).expect("known dtype");
            rows.push(json!({
                "name": s.name, "dtype": s.data.dtype(), "shape": s.shape,
                "offset": off, "bytes": bytes,
            }));
            off += bytes;
        }
        (rows, off)
    };
    let text_for = |rows: Vec<Value>| -> String {
        json!({
            "index": index, "n": n, "dim": dim, "build_params": params_to_json(build_params),
            "seed": seed, "contract_version": 1, "language": "rust", "sections": rows,
        })
        .to_string()
    };
    // Offsets depend on the header length and the header length on the offsets' digits.
    let mut base = align(12);
    let mut found = None;
    for _ in 0..16 {
        let (rows, end) = table(base);
        let text = text_for(rows);
        let need = align(12 + text.len() as u64);
        if need == base {
            found = Some((text, end));
            break;
        }
        base = need.max(base);
    }
    let (text, end) = found.ok_or("vro: header size did not converge")?;
    if !text.is_ascii() {
        return Err("vro: header is not ASCII".into());
    }

    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut w = BufWriter::with_capacity(CHUNK_BYTES, file);
    let io = |e: std::io::Error| format!("{}: {e}", path.display());
    w.write_all(MAGIC).map_err(io)?;
    w.write_all(&(text.len() as u32).to_le_bytes()).map_err(io)?;
    w.write_all(text.as_bytes()).map_err(io)?;
    let mut pos = 12 + text.len() as u64;
    let mut buf: Vec<u8> = Vec::with_capacity(CHUNK_BYTES);
    for s in sections {
        let off = align(pos);
        w.write_all(&vec![0u8; (off - pos) as usize]).map_err(io)?;
        pos = off;
        match &s.data {
            Data::U8(d) => {
                w.write_all(d).map_err(io)?;
                pos += d.len() as u64;
            }
            Data::F32(d) => {
                for chunk in d.chunks(CHUNK_BYTES / 4) {
                    buf.clear();
                    chunk.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes()));
                    w.write_all(&buf).map_err(io)?;
                }
                pos += d.len() as u64 * 4;
            }
            Data::I32(d) => {
                for chunk in d.chunks(CHUNK_BYTES / 4) {
                    buf.clear();
                    chunk.iter().for_each(|x| buf.extend_from_slice(&x.to_le_bytes()));
                    w.write_all(&buf).map_err(io)?;
                }
                pos += d.len() as u64 * 4;
            }
        }
    }
    debug_assert_eq!(pos, end);
    let file = w.into_inner().map_err(|e| io(e.into_error()))?;
    file.sync_all().map_err(io)?;
    Ok(end)
}

/// Parses the header text and checks the section table against the file size.
fn parse_header(text: &str, header_len: u32, file_len: u64) -> Result<Header, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("vro: header JSON: {e}"))?;
    let str_of = |k: &str| -> Result<String, String> {
        v.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("vro: header has no string '{k}'"))
    };
    let u64_of = |k: &str| -> Result<u64, String> {
        v.get(k)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("vro: header has no non-negative integer '{k}'"))
    };
    let build_params = params_from_json(v.get("build_params").ok_or("vro: header has no build_params")?)?;
    let rows = v
        .get("sections")
        .and_then(Value::as_array)
        .ok_or("vro: header has no sections array")?;
    let mut sections = Vec::with_capacity(rows.len());
    let mut end = align(12 + header_len as u64);
    for r in rows {
        let name = r.get("name").and_then(Value::as_str).ok_or("vro: section without name")?;
        let dtype = r.get("dtype").and_then(Value::as_str).ok_or("vro: section without dtype")?;
        let size = dtype_size(dtype).ok_or_else(|| format!("vro: section {name}: unknown dtype {dtype}"))?;
        let shape: Vec<usize> = r
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("vro: section {name} without shape"))?
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("vro: section {name}: bad shape"))?;
        let offset = r.get("offset").and_then(Value::as_u64).ok_or("vro: section without offset")?;
        let bytes = r.get("bytes").and_then(Value::as_u64).ok_or("vro: section without bytes")?;
        if offset % ALIGN != 0 {
            return Err(format!("vro: section {name}: offset {offset} is not a multiple of 64"));
        }
        if offset < end {
            return Err(format!("vro: section {name}: offset {offset} overlaps earlier data"));
        }
        let want = shape.iter().product::<usize>() as u64 * size;
        if bytes != want {
            return Err(format!("vro: section {name}: {bytes} bytes, shape {shape:?} needs {want}"));
        }
        if offset + bytes > file_len {
            return Err(format!("vro: section {name} ends after the end of the file"));
        }
        if sections.iter().any(|s: &SectionInfo| s.name == name) {
            return Err(format!("vro: section {name} appears twice"));
        }
        end = offset + bytes;
        sections.push(SectionInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
            offset,
            bytes,
        });
    }
    let h = Header {
        index: str_of("index")?,
        n: u64_of("n")? as usize,
        dim: u64_of("dim")? as usize,
        build_params,
        seed: u64_of("seed")?,
        contract_version: u64_of("contract_version")?,
        language: str_of("language")?,
        sections,
        header_len,
    };
    if h.contract_version != 1 {
        return Err(format!("vro: contract_version {} (expected 1)", h.contract_version));
    }
    Ok(h)
}

/// An open `.vro` file: the parsed header and typed section readers.
pub struct Reader {
    file: File,
    pub header: Header,
    path: String,
}

impl Reader {
    /// Opens `path`, checks the magic, parses the header and the section table.
    pub fn open(path: &Path) -> Result<Self, String> {
        let p = path.display().to_string();
        let mut file = File::open(path).map_err(|e| format!("{p}: {e}"))?;
        let file_len = file.metadata().map_err(|e| format!("{p}: {e}"))?.len();
        let mut head = [0u8; 12];
        file.read_exact(&mut head).map_err(|_| format!("{p}: file too short for a .vro header"))?;
        if &head[..8] != MAGIC {
            return Err(format!("{p}: not a .vro file (magic is not VROIDX01)"));
        }
        let header_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]);
        if header_len > MAX_HEADER || 12 + header_len as u64 > file_len {
            return Err(format!("{p}: bad header_len {header_len}"));
        }
        let mut text = vec![0u8; header_len as usize];
        file.read_exact(&mut text).map_err(|e| format!("{p}: {e}"))?;
        if !text.is_ascii() {
            return Err(format!("{p}: header is not ASCII"));
        }
        let text = String::from_utf8(text).expect("ASCII is UTF-8");
        let header = parse_header(&text, header_len, file_len).map_err(|e| format!("{p}: {e}"))?;
        Ok(Self { file, header, path: p })
    }

    /// Raw bytes of section `name`, converted chunk by chunk with `conv`.
    fn read_as<T>(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        size: usize,
        conv: impl Fn(&[u8]) -> T,
    ) -> Result<Vec<T>, String> {
        let s = self.header.section(name)?.clone();
        if s.dtype != dtype {
            return Err(format!("{}: section {name} has dtype {}, expected {dtype}", self.path, s.dtype));
        }
        if s.shape != shape {
            return Err(format!(
                "{}: section {name} has shape {:?}, expected {shape:?}",
                self.path, s.shape
            ));
        }
        self.file
            .seek(SeekFrom::Start(s.offset))
            .map_err(|e| format!("{}: {e}", self.path))?;
        let mut out = Vec::with_capacity(s.bytes as usize / size);
        let mut left = s.bytes as usize;
        let mut buf = vec![0u8; CHUNK_BYTES.min(left.max(1))];
        while left > 0 {
            let take = left.min(buf.len());
            self.file
                .read_exact(&mut buf[..take])
                .map_err(|e| format!("{}: section {name}: {e}", self.path))?;
            out.extend(buf[..take].chunks_exact(size).map(&conv));
            left -= take;
        }
        Ok(out)
    }

    pub fn read_f32(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        self.read_as(name, "f32", shape, 4, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_i32(&mut self, name: &str, shape: &[usize]) -> Result<Vec<i32>, String> {
        self.read_as(name, "int32", shape, 4, |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u8(&mut self, name: &str, shape: &[usize]) -> Result<Vec<u8>, String> {
        self.read_as(name, "u8", shape, 1, |b| b[0])
    }

    /// Shape of section `name` from the table.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>, String> {
        Ok(self.header.section(name)?.shape.clone())
    }

    /// The `vectors` section as a matrix (n x dim of the header).
    pub fn read_vectors(&mut self) -> Result<crate::Matrix, String> {
        let (n, dim) = (self.header.n, self.header.dim);
        let data = self.read_f32("vectors", &[n, dim])?;
        Ok(crate::Matrix { data, rows: n, cols: dim })
    }

    /// The `tombstones` section. `None` when no bit is set.
    pub fn read_tombstones(&mut self) -> Result<Option<Tombstones>, String> {
        let n = self.header.n;
        let bytes = self.read_u8("tombstones", &[n.div_ceil(8)])?;
        let t = Tombstones::from_bytes(&bytes, n);
        Ok((t.deleted() > 0).then_some(t))
    }
}

/// Reads and checks only the header of `path`.
pub fn read_header(path: &Path) -> Result<Header, String> {
    Reader::open(path).map(|r| r.header)
}

/// The tombstone bit set as bytes: bit i (LSB first within a byte) = row i deleted.
pub fn tombstone_bytes(t: Option<&Tombstones>, n: usize) -> Vec<u8> {
    t.map_or_else(|| vec![0u8; n.div_ceil(8)], Tombstones::to_bytes)
}
