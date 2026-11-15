// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text to syntax tree: a lexer and a Pratt parser.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::Span;
use crate::error::{Error, ErrorKind};

/// How deep a tree may nest. Every later pass recurses over the tree, so
/// this bounds their stack use whatever the input.
pub(crate) const MAX_DEPTH: usize = 64;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Token<'s> {
    /// A number's text, and whether it has a fraction or an exponent.
    Number(&'s str, bool),
    Name(&'s str),
    Punct(&'static str),
    End,
}

/// Every operator and delimiter, longest first so `<=` wins over `<`.
const PUNCTS: &[&str] = &[
    "<=", ">=", "==", "!=", "&&", "||", "+", "-", "*", "/", "<", ">", "!", "?", ":", "(", ")", ",",
];

struct Lexer<'s> {
    source: &'s str,
    at: usize,
}

impl<'s> Lexer<'s> {
    fn skip_space(&mut self) {
        let rest = &self.source.as_bytes()[self.at..];
        self.at += rest
            .iter()
            .take_while(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
            .count();
    }

    fn digits(&self, from: usize) -> usize {
        self.source.as_bytes()[from..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count()
    }

    /// The next token and its span.
    fn next(&mut self) -> Result<(Token<'s>, Span), Error> {
        self.skip_space();
        let start = self.at;
        let bytes = self.source.as_bytes();
        let Some(&first) = bytes.get(start) else {
            return Ok((Token::End, Span::new(start, start)));
        };
        if first.is_ascii_digit() {
            // digits ( "." digits )? ( ("e" | "E") ("+" | "-")? digits )?
            let mut end = start + self.digits(start);
            let mut real = false;
            if bytes.get(end) == Some(&b'.') {
                let n = self.digits(end + 1);
                if n == 0 {
                    return Err(Error::new(
                        Span::new(start, end + 1),
                        ErrorKind::Syntax("digits after the decimal point"),
                    ));
                }
                end += 1 + n;
                real = true;
            }
            if matches!(bytes.get(end), Some(b'e' | b'E')) {
                let mut at = end + 1;
                if matches!(bytes.get(at), Some(b'+' | b'-')) {
                    at += 1;
                }
                let n = self.digits(at);
                if n == 0 {
                    return Err(Error::new(
                        Span::new(start, at),
                        ErrorKind::Syntax("digits in the exponent"),
                    ));
                }
                end = at + n;
                real = true;
            }
            if bytes
                .get(end)
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'.')
            {
                return Err(Error::new(
                    Span::new(start, end + 1),
                    ErrorKind::Syntax("a number"),
                ));
            }
            self.at = end;
            return Ok((
                Token::Number(&self.source[start..end], real),
                Span::new(start, end),
            ));
        }
        if first.is_ascii_alphabetic() || first == b'_' {
            let n = bytes[start..]
                .iter()
                .take_while(|b| b.is_ascii_alphanumeric() || **b == b'_')
                .count();
            self.at = start + n;
            return Ok((
                Token::Name(&self.source[start..start + n]),
                Span::new(start, start + n),
            ));
        }
        for p in PUNCTS {
            if bytes[start..].starts_with(p.as_bytes()) {
                self.at = start + p.len();
                return Ok((Token::Punct(p), Span::new(start, self.at)));
            }
        }
        // Report the whole character, whatever its width.
        let width = self.source[start..]
            .chars()
            .next()
            .map_or(1, char::len_utf8);
        Err(Error::new(
            Span::new(start, start + width),
            ErrorKind::Syntax("a number, a name, an operator or a parenthesis"),
        ))
    }
}

/// A unary operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Unary {
    /// `-`
    Neg,
    /// `!`
    Not,
}

/// A binary operator.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl Binary {
    fn of(p: &str) -> Option<Self> {
        Some(match p {
            "+" => Self::Add,
            "-" => Self::Sub,
            "*" => Self::Mul,
            "/" => Self::Div,
            "<" => Self::Lt,
            ">" => Self::Gt,
            "<=" => Self::Le,
            ">=" => Self::Ge,
            "==" => Self::Eq,
            "!=" => Self::Ne,
            "&&" => Self::And,
            "||" => Self::Or,
            _ => return None,
        })
    }

    /// Left and right binding power. Higher binds tighter; a right power
    /// above the left makes the operator left-associative.
    const fn power(self) -> (u8, u8) {
        match self {
            Self::Or => (1, 2),
            Self::And => (3, 4),
            Self::Lt | Self::Gt | Self::Le | Self::Ge | Self::Eq | Self::Ne => (5, 6),
            Self::Add | Self::Sub => (7, 8),
            Self::Mul | Self::Div => (9, 10),
        }
    }

    pub(crate) const fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Lt | Self::Gt | Self::Le | Self::Ge | Self::Eq | Self::Ne
        )
    }

    /// Its spelling, for messages.
    pub(crate) const fn symbol(self) -> &'static str {
        match self {
            Self::Add => "`+`",
            Self::Sub => "`-`",
            Self::Mul => "`*`",
            Self::Div => "`/`",
            Self::Lt => "`<`",
            Self::Gt => "`>`",
            Self::Le => "`<=`",
            Self::Ge => "`>=`",
            Self::Eq => "`==`",
            Self::Ne => "`!=`",
            Self::And => "`&&`",
            Self::Or => "`||`",
        }
    }
}

/// A syntax tree node: what it is, and where.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Expr<'s> {
    pub(crate) kind: ExprKind<'s>,
    pub(crate) span: Span,
    /// The depth of the tree below and including this node.
    depth: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExprKind<'s> {
    /// A number's text, and whether it has a fraction or an exponent.
    Number(&'s str, bool),
    Bool(bool),
    Name(&'s str),
    Unary(Unary, Box<Expr<'s>>),
    Binary(Binary, Box<Expr<'s>>, Box<Expr<'s>>),
    /// `cond ? then : otherwise`
    Cond(Box<Expr<'s>>, Box<Expr<'s>>, Box<Expr<'s>>),
    /// A function's name and span, and the arguments.
    Call(&'s str, Span, Vec<Expr<'s>>),
}

impl<'s> Expr<'s> {
    fn node(kind: ExprKind<'s>, span: Span) -> Result<Self, Error> {
        let below = match &kind {
            ExprKind::Number(..) | ExprKind::Bool(_) | ExprKind::Name(_) => 0,
            ExprKind::Unary(_, a) => a.depth,
            ExprKind::Binary(_, a, b) => a.depth.max(b.depth),
            ExprKind::Cond(a, b, c) => a.depth.max(b.depth).max(c.depth),
            ExprKind::Call(_, _, args) => args.iter().map(|a| a.depth).max().unwrap_or(0),
        };
        let depth = 1 + below;
        if depth > MAX_DEPTH {
            return Err(Error::new(span, ErrorKind::TooDeep));
        }
        Ok(Self { kind, span, depth })
    }
}

struct Parser<'s> {
    lexer: Lexer<'s>,
    token: Token<'s>,
    span: Span,
    /// Open parentheses, `?:` and unary operators: recursion the tree's
    /// depth does not see until a node is made.
    nesting: usize,
}

impl<'s> Parser<'s> {
    fn bump(&mut self) -> Result<(), Error> {
        (self.token, self.span) = self.lexer.next()?;
        Ok(())
    }

    fn eat(&mut self, p: &'static str) -> Result<bool, Error> {
        if self.token == Token::Punct(p) {
            self.bump()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expect(&mut self, p: &'static str, what: &'static str) -> Result<Span, Error> {
        let span = self.span;
        if self.eat(p)? {
            Ok(span)
        } else {
            Err(Error::new(self.span, ErrorKind::Syntax(what)))
        }
    }

    fn enter(&mut self) -> Result<(), Error> {
        self.nesting += 1;
        if self.nesting > MAX_DEPTH {
            return Err(Error::new(self.span, ErrorKind::TooDeep));
        }
        Ok(())
    }

    /// `or ( "?" expr ":" expr )?`
    fn expr(&mut self) -> Result<Expr<'s>, Error> {
        self.enter()?;
        let cond = self.binary(0)?;
        let e = if self.eat("?")? {
            let then = self.expr()?;
            self.expect(":", "`:`")?;
            let otherwise = self.expr()?;
            let span = cond.span.to(otherwise.span);
            let kind = ExprKind::Cond(Box::new(cond), Box::new(then), Box::new(otherwise));
            Expr::node(kind, span)?
        } else {
            cond
        };
        self.nesting -= 1;
        Ok(e)
    }

    /// Binary operators binding at least as tightly as `min`.
    fn binary(&mut self, min: u8) -> Result<Expr<'s>, Error> {
        let mut lhs = self.unary()?;
        // Whether `lhs` is a comparison made at this level: another
        // comparison would chain.
        let mut compared = false;
        while let Token::Punct(p) = self.token
            && let Some(op) = Binary::of(p)
        {
            let (left, right) = op.power();
            if left < min {
                break;
            }
            let op_span = self.span;
            if op.is_comparison() && compared {
                return Err(Error::new(op_span, ErrorKind::ChainedComparison));
            }
            self.bump()?;
            let rhs = self.binary(right)?;
            compared = op.is_comparison();
            let span = lhs.span.to(rhs.span);
            lhs = Expr::node(ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)), span)?;
        }
        Ok(lhs)
    }

    /// `("-" | "!") unary | primary`
    fn unary(&mut self) -> Result<Expr<'s>, Error> {
        let op = match self.token {
            Token::Punct("-") => Unary::Neg,
            Token::Punct("!") => Unary::Not,
            _ => return self.primary(),
        };
        let start = self.span;
        self.bump()?;
        self.enter()?;
        let operand = self.unary()?;
        self.nesting -= 1;
        let span = start.to(operand.span);
        Expr::node(ExprKind::Unary(op, Box::new(operand)), span)
    }

    fn primary(&mut self) -> Result<Expr<'s>, Error> {
        let span = self.span;
        match self.token {
            Token::Number(text, real) => {
                self.bump()?;
                Expr::node(ExprKind::Number(text, real), span)
            }
            Token::Name("true") => {
                self.bump()?;
                Expr::node(ExprKind::Bool(true), span)
            }
            Token::Name("false") => {
                self.bump()?;
                Expr::node(ExprKind::Bool(false), span)
            }
            Token::Name(name) => {
                self.bump()?;
                if !self.eat("(")? {
                    return Expr::node(ExprKind::Name(name), span);
                }
                self.enter()?;
                let mut args = Vec::new();
                if self.token != Token::Punct(")") {
                    loop {
                        args.push(self.expr()?);
                        if !self.eat(",")? {
                            break;
                        }
                    }
                }
                let close = self.expect(")", "`,` or `)`")?;
                self.nesting -= 1;
                Expr::node(ExprKind::Call(name, span, args), span.to(close))
            }
            Token::Punct("(") => {
                self.bump()?;
                self.enter()?;
                let mut inner = self.expr()?;
                let close = self.expect(")", "`)`")?;
                self.nesting -= 1;
                inner.span = span.to(close);
                Ok(inner)
            }
            _ => Err(Error::new(
                span,
                ErrorKind::Syntax("a number, a name or `(`"),
            )),
        }
    }
}

/// Parses `source` as one expression.
pub(crate) fn parse(source: &str) -> Result<Expr<'_>, Error> {
    let mut parser = Parser {
        lexer: Lexer { source, at: 0 },
        token: Token::End,
        span: Span::default(),
        nesting: 0,
    };
    parser.bump()?;
    let e = parser.expr()?;
    if parser.token != Token::End {
        return Err(Error::new(
            parser.span,
            ErrorKind::Syntax("an operator or the end"),
        ));
    }
    Ok(e)
}
