// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Machine-readable reports: named measurements and checks, as JSON.
//!
//! A [`Report`] knows nothing of materials: it is a subject and a list of
//! entries, each a [`Measurement`] (a number with a unit and, optionally,
//! the bounds it must lie in) or a [`Check`] (a named pass or fail with a
//! detail). Any harness that measures something (a material bake, a tree
//! generator's reference renders) can fill one and write it with
//! [`Report::to_json`], which is deterministic: entries keep their order,
//! and numbers print with Rust's shortest round-tripping formatting.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

/// A number measured, with its unit and the bounds it must lie in.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// A dotted name, such as `"base_color.min"`.
    pub name: String,
    /// The value.
    pub value: f64,
    /// Its unit, such as `"m"`, or `""`.
    pub unit: &'static str,
    /// The closed range it must lie in, if any.
    pub bounds: Option<[f64; 2]>,
}

impl Measurement {
    /// Whether the value is finite and within its bounds.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.value.is_finite()
            && self
                .bounds
                .is_none_or(|[lo, hi]| lo <= self.value && self.value <= hi)
    }
}

/// A named pass or fail.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// A dotted name, such as `"seam.height.x"`.
    pub name: String,
    /// Whether it passed.
    pub pass: bool,
    /// What was found.
    pub detail: String,
}

/// One entry of a [`Report`].
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    /// A measurement.
    Measurement(Measurement),
    /// A check.
    Check(Check),
}

/// A report about one subject: see the [module docs](self).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// What was measured.
    pub subject: String,
    /// The entries, in the order they were made.
    pub entries: Vec<Entry>,
}

impl Report {
    /// An empty report about `subject`.
    #[must_use]
    pub fn new(subject: &str) -> Self {
        Self {
            subject: subject.into(),
            entries: Vec::new(),
        }
    }

    /// Records a measurement.
    pub fn measure(
        &mut self,
        name: &str,
        value: f64,
        unit: &'static str,
        bounds: Option<[f64; 2]>,
    ) {
        self.entries.push(Entry::Measurement(Measurement {
            name: name.into(),
            value,
            unit,
            bounds,
        }));
    }

    /// Records a check.
    pub fn check(&mut self, name: &str, pass: bool, detail: impl Into<String>) {
        self.entries.push(Entry::Check(Check {
            name: name.into(),
            pass,
            detail: detail.into(),
        }));
    }

    /// Appends another report's entries, their names prefixed with
    /// `prefix` and a dot.
    pub fn merge(&mut self, prefix: &str, other: Self) {
        for mut entry in other.entries {
            match &mut entry {
                Entry::Measurement(m) => m.name = format!("{prefix}.{}", m.name),
                Entry::Check(c) => c.name = format!("{prefix}.{}", c.name),
            }
            self.entries.push(entry);
        }
    }

    /// The entries that fail: checks that did not pass and measurements
    /// outside their bounds or not finite.
    pub fn failures(&self) -> impl Iterator<Item = &Entry> + '_ {
        self.entries.iter().filter(|e| match e {
            Entry::Measurement(m) => !m.passes(),
            Entry::Check(c) => !c.pass,
        })
    }

    /// Whether nothing failed.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.failures().next().is_none()
    }

    /// The report as JSON:
    /// `{"subject": …, "passed": …, "measurements": […], "checks": […]}`.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        let _ = write!(
            out,
            "{{\n  \"subject\": {},\n  \"passed\": {},\n  \"measurements\": [",
            quote(&self.subject),
            self.passed()
        );
        let mut first = true;
        for e in &self.entries {
            if let Entry::Measurement(m) = e {
                let bounds = m.bounds.map_or_else(
                    || String::from("null"),
                    |[lo, hi]| format!("[{}, {}]", number(lo), number(hi)),
                );
                let _ = write!(
                    out,
                    "{}\n    {{\"name\": {}, \"value\": {}, \"unit\": {}, \"bounds\": {}, \"pass\": {}}}",
                    if first { "" } else { "," },
                    quote(&m.name),
                    number(m.value),
                    quote(m.unit),
                    bounds,
                    m.passes()
                );
                first = false;
            }
        }
        out.push_str("\n  ],\n  \"checks\": [");
        first = true;
        for e in &self.entries {
            if let Entry::Check(c) = e {
                let _ = write!(
                    out,
                    "{}\n    {{\"name\": {}, \"pass\": {}, \"detail\": {}}}",
                    if first { "" } else { "," },
                    quote(&c.name),
                    c.pass,
                    quote(&c.detail)
                );
                first = false;
            }
        }
        out.push_str("\n  ]\n}\n");
        out
    }
}

/// A JSON number; non-finite values become `null`.
fn number(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        String::from("null")
    }
}

/// A JSON string.
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_serialize_deterministically() {
        let mut r = Report::new("brick \"wall\"");
        r.measure("height.max", 0.006, "m", Some([0.0, 0.01]));
        r.measure("nan", f64::NAN, "", None);
        r.check("seam.x", true, "ratio 1.1");
        assert!(!r.passed(), "the NaN fails");
        assert_eq!(r.failures().count(), 1);
        let json = r.to_json();
        assert!(
            json.contains("\"subject\": \"brick \\\"wall\\\"\""),
            "{json}"
        );
        assert!(json.contains("\"value\": null"), "{json}");
        assert!(json.contains("\"bounds\": [0, 0.01]"), "{json}");
        assert_eq!(json, r.to_json(), "deterministic");
    }
}
