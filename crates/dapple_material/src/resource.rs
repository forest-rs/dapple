// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Host-resolved resource inputs: images and exemplars a module samples.
//!
//! A module declares what it needs ([`ResourceRequest`]: a semantic type
//! and whether the image must tile); an instance names a logical resource
//! ([`ResourceRef`]); the host resolves the name ([`ResourceHost`]) into a
//! [`Resolved`] image. Decoding, color conversion and file access happen in
//! the host, outside this `no_std` crate. What reaches the module is:
//!
//! - **content identity**: a fingerprint of the decoded texels, which the
//!   host computes and dapple uses as the image's derivation, so a changed
//!   file changes every result that read it and an unchanged one does not;
//! - **semantic type** and **color information**: a
//!   [`SampleImage`]'s [`PortType`], linear, with its primaries;
//! - **physical scale**: the image's texel size in meters, so a 20 cm
//!   exemplar stays 20 cm on any grid;
//! - **mip policy**: the [`ReductionPolicy`] its levels were built with,
//!   checked against its type.

use alloc::string::String;
use core::fmt;

use dapple_field::program::Fingerprint;
use dapple_field::{Domain, PortType, SampleImage};
use dapple_raster::typed::ReductionPolicy;

/// What a module needs from a resource.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ResourceRequest {
    /// The semantic type of the texels.
    pub port: PortType,
    /// Whether the image must be periodic, to tile.
    pub periodic: bool,
}

/// A logical resource name, resolved by the host.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceRef(pub String);

/// A resolved resource.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    /// The decoded texels' content fingerprint, computed by the host.
    pub content: Fingerprint,
    /// The image, in meters, linear, typed, with its levels.
    pub image: SampleImage,
    /// How its coarser levels were reduced.
    pub mips: ReductionPolicy,
}

/// Why a resource could not be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceError {
    /// The host has no such resource.
    NotFound(ResourceRef),
    /// The resolved image's type is not the requested one.
    WrongType {
        /// The resource.
        reference: ResourceRef,
        /// What the module asked for.
        requested: PortType,
        /// What the host supplied.
        found: PortType,
    },
    /// The module needs a tiling image and the host supplied a clamping
    /// one.
    NotPeriodic(ResourceRef),
    /// The mip policy does not fit the image's type.
    MipPolicy(ResourceRef),
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(r) => write!(f, "no resource {:?}", r.0),
            Self::WrongType {
                reference,
                requested,
                found,
            } => write!(f, "{:?} holds {found}, not {requested}", reference.0),
            Self::NotPeriodic(r) => write!(f, "{:?} does not tile", r.0),
            Self::MipPolicy(r) => write!(f, "{:?}: mip policy does not fit its type", r.0),
        }
    }
}

impl core::error::Error for ResourceError {}

/// Resolves logical resources for module instances.
pub trait ResourceHost {
    /// The resource `reference` names.
    ///
    /// # Errors
    ///
    /// [`ResourceError::NotFound`] when there is none.
    fn resolve(&self, reference: &ResourceRef) -> Result<Resolved, ResourceError>;
}

/// A host with no resources.
#[derive(Copy, Clone, Debug, Default)]
pub struct NoResources;

impl ResourceHost for NoResources {
    fn resolve(&self, reference: &ResourceRef) -> Result<Resolved, ResourceError> {
        Err(ResourceError::NotFound(reference.clone()))
    }
}

/// Checks `resolved` against `request`.
///
/// # Errors
///
/// [`ResourceError::WrongType`], [`ResourceError::NotPeriodic`] or
/// [`ResourceError::MipPolicy`].
pub fn check(
    reference: &ResourceRef,
    request: ResourceRequest,
    resolved: &Resolved,
) -> Result<(), ResourceError> {
    let found = resolved.image.port();
    if found != request.port {
        return Err(ResourceError::WrongType {
            reference: reference.clone(),
            requested: request.port,
            found,
        });
    }
    if request.periodic && !matches!(resolved.image.domain(), Domain::Periodic { .. }) {
        return Err(ResourceError::NotPeriodic(reference.clone()));
    }
    if resolved.mips.check(found).is_err() {
        return Err(ResourceError::MipPolicy(reference.clone()));
    }
    Ok(())
}
