//! A hand-written reader for NumPy `.npy` files, format version 1.0 (CONTRACT 1.1).

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const MAGIC: &[u8; 6] = b"\x93NUMPY";
/// Bytes converted per read, so a large file never needs a second full-size buffer.
const CHUNK_BYTES: usize = 1 << 22;

/// A row-major 2-D array in one contiguous buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct Array2<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

/// A float32 matrix: the corpus or the queries.
pub type Matrix = Array2<f32>;

impl<T> Array2<T> {
    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

/// The parsed header fields that the reader checks.
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
    /// Offset of the first data byte: 10 + HLEN.
    pub data_offset: u64,
}

/// Reads a `<f4` 2-D array. `max_rows` limits the rows read (`--limit`).
pub fn read_f32(path: &Path, max_rows: Option<usize>) -> Result<Matrix, String> {
    read_array(path, "<f4", max_rows, |b: &[u8; 4]| {
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    })
}

/// Reads a `<i8` (int64) 2-D array, for example `ground_truth.npy`.
pub fn read_i64(path: &Path, max_rows: Option<usize>) -> Result<Array2<i64>, String> {
    read_array(path, "<i8", max_rows, |b: &[u8; 8]| {
        i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    })
}

fn read_array<T, const W: usize>(
    path: &Path,
    descr: &str,
    max_rows: Option<usize>,
    convert: impl Fn(&[u8; W]) -> T,
) -> Result<Array2<T>, String> {
    let err = |msg: String| format!("{}: {msg}", path.display());
    let file = File::open(path).map_err(|e| err(e.to_string()))?;
    let mut reader = BufReader::new(file);
    let header = read_header(&mut reader).map_err(err)?;
    if header.descr != descr {
        return Err(err(format!(
            "descr is '{}', expected '{descr}'",
            header.descr
        )));
    }
    let (total_rows, cols) = match header.shape[..] {
        [r, c] => (r, c),
        _ => return Err(err(format!("shape {:?} is not 2-D", header.shape))),
    };
    let rows = max_rows.map_or(total_rows, |m| m.min(total_rows));
    reader
        .seek(SeekFrom::Start(header.data_offset))
        .map_err(|e| err(e.to_string()))?;
    let data = read_values(&mut reader, rows * cols, convert).map_err(err)?;
    Ok(Array2 { data, rows, cols })
}

/// Reads `count` little-endian values of `W` bytes each, in chunks.
fn read_values<T, const W: usize>(
    reader: &mut impl Read,
    count: usize,
    convert: impl Fn(&[u8; W]) -> T,
) -> Result<Vec<T>, String> {
    let mut out = Vec::with_capacity(count);
    let mut buf = vec![0u8; CHUNK_BYTES - CHUNK_BYTES % W];
    let mut left = count * W;
    while left > 0 {
        let n = left.min(buf.len());
        reader
            .read_exact(&mut buf[..n])
            .map_err(|e| format!("data is shorter than the shape says: {e}"))?;
        out.extend(buf[..n].chunks_exact(W).map(|c| {
            let bytes: &[u8; W] = c.try_into().expect("chunk has W bytes");
            convert(bytes)
        }));
        left -= n;
    }
    Ok(out)
}

/// Reads and parses the magic string, version, and header dict.
pub fn read_header(reader: &mut impl Read) -> Result<Header, String> {
    let mut pre = [0u8; 10];
    reader
        .read_exact(&mut pre)
        .map_err(|e| format!("cannot read header: {e}"))?;
    if &pre[..6] != MAGIC {
        return Err("not a .npy file (bad magic string)".into());
    }
    if pre[6] != 1 {
        return Err(format!(
            "npy version {}.{} is not supported, need 1.0",
            pre[6], pre[7]
        ));
    }
    let hlen = u16::from_le_bytes([pre[8], pre[9]]) as usize;
    let mut dict = vec![0u8; hlen];
    reader
        .read_exact(&mut dict)
        .map_err(|e| format!("cannot read header dict: {e}"))?;
    let dict = String::from_utf8(dict).map_err(|_| "header dict is not ASCII".to_string())?;
    parse_dict(&dict, 10 + hlen as u64)
}

/// Parses `{'descr': '<f4', 'fortran_order': False, 'shape': (N, D), }`.
fn parse_dict(dict: &str, data_offset: u64) -> Result<Header, String> {
    let descr = value_after(dict, "descr")?;
    let descr = descr
        .strip_prefix('\'')
        .and_then(|s| s.split('\'').next())
        .ok_or_else(|| format!("bad descr in header: {dict}"))?
        .to_string();
    let fortran = value_after(dict, "fortran_order")?;
    let fortran_order = if fortran.starts_with("False") {
        false
    } else if fortran.starts_with("True") {
        return Err("fortran_order is True; only C order is supported".into());
    } else {
        return Err(format!("bad fortran_order in header: {dict}"));
    };
    let shape = value_after(dict, "shape")?;
    let inner = shape
        .strip_prefix('(')
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| format!("bad shape in header: {dict}"))?;
    let shape = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("bad shape entry '{s}'"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Header {
        descr,
        fortran_order,
        shape,
        data_offset,
    })
}

/// Returns the text after `'key':`, with leading spaces removed.
fn value_after<'a>(dict: &'a str, key: &str) -> Result<&'a str, String> {
    let pat = format!("'{key}':");
    let start = dict
        .find(&pat)
        .ok_or_else(|| format!("header has no '{key}': {dict}"))?;
    Ok(dict[start + pat.len()..].trim_start())
}
