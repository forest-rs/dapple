// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Solid oak baked onto exedra timbers through their texture charts.
//!
//! A post, a beam across it and a rafter with angled end cuts are built as
//! exedra extrusions with construction charts. Every face region gets its
//! own material slot and its own texture, baked by `dapple_exedra` from
//! `dapple_library`'s solid oak: each texel evaluates the wood at the
//! surface point it stands for, so the end cuts show rings and rays and the
//! long faces show grain. Each timber sits in its own place in the log: the
//! post is boxed heart, the beam flat sawn and the rafter off-center.
//! Texels are filtered anisotropically along their footprint on the
//! surface (`SolidProgram::eval_chart_anisotropic`), so a chart stretched
//! over a face is filtered along its length without blurring across it.
//! The wood's rays are a `fract` of the angle around the pith, which no
//! footprint evaluation can filter: the program's sampling guarantee is
//! point-only, so each texel is integrated by stratified point samples
//! instead, and the rays stay clean rather than aliasing into stair steps.
//! Chart textures are baked for one region each and never tile, so they
//! declare no periodicity and have no seam to check.
//!
//! Run with `cargo run -p timber_bake --release -- [output-dir]`; the default
//! output directory is the repository's git-ignored `.local/gallery/timber-bake`.
//! It writes `timbers.glb` (glTF with `KHR_texture_transform` placing each
//! region's UVs on its texture) and each texture as a PNG. Render it with
//! `blender --background --python examples/timber_bake/tools/render.py -- <output-dir>`.

use std::collections::BTreeMap;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use dapple_exedra::{BakeParams, SurfaceBake};
use dapple_library::oak;
use exedra_assembly::{Assembly, CompilePolicy, PartCompiler, PartId};
use exedra_constructive::builders;
use exedra_constructive::chart::{ChartTransform, SurfaceChart};
use exedra_constructive::ir::{CapMode, NodeKind, Placement3, Recipe, RecipeBuilder};
use exedra_constructive::profile::{Loop2, Profile2, Seg2};
use exedra_gltf::{GltfExportOptions, MaterialResolver, Texture, export_glb_with_materials};
use glam::{Affine3A, Vec3};
use serde_json::{Value, json};

/// Texels per meter of surface: 2 mm texels.
const TEXELS_PER_METER: f32 = 500.0;

/// One timber: its profile and extrusion, and where it sits in the log.
struct Timber {
    key: &'static str,
    recipe: Recipe,
    /// Regions of the extrusion: two caps and one wall per profile segment.
    regions: u32,
    /// Maps the part's own coordinates into the wood, whose trunk runs along
    /// z through the origin.
    in_log: Affine3A,
    placement: Placement3,
}

/// An extrusion of `profile` along its local z by `length`, charted, with a
/// material slot per region and no authored body slot, so regions bind to
/// slots through the assembly.
fn extrusion(profile: Profile2, length: f64) -> (Recipe, u32) {
    let segments = u32::try_from(profile.outer().segs().len()).expect("small profile");
    let regions = 2 + segments;
    let mut b = RecipeBuilder::new();
    for region in 0..regions {
        b.material_slot(&format!("r{region}"));
    }
    let profile = b.add_profile(profile);
    let root = b
        .with_surface_chart(SurfaceChart::Extrude {
            wall: ChartTransform::IDENTITY,
            caps: ChartTransform::IDENTITY,
        })
        .add(NodeKind::Extrude {
            profile,
            placement: Placement3::IDENTITY,
            height: length,
            caps: CapMode::Both,
        })
        .expect("valid extrusion");
    (b.finish(root).expect("valid recipe"), regions)
}

fn timbers() -> Vec<Timber> {
    let quarter = std::f64::consts::FRAC_PI_2;
    // A 16 cm square post, 1.2 m tall, cut around the pith: end grain shows
    // the whole ring pattern, the faces show flat and quarter grain.
    let (post, post_regions) = extrusion(builders::rect_centered(0.16, 0.16).unwrap(), 1.2);
    // A 15 × 20 cm beam, 1.6 m long, sawn 12 cm from the pith: its top and
    // bottom are flat sawn, its sides nearly quarter sawn.
    let (beam, beam_regions) = extrusion(builders::rect_centered(0.15, 0.2).unwrap(), 1.6);
    // A rafter 1.1 m long and 12 cm deep, its ends cut at a 35 degree plumb
    // angle, extruded 7 cm thick. Its long axis is the profile's x.
    let shear = 0.12 * 35.0_f64.to_radians().tan();
    let rafter_profile = Profile2::simple(
        Loop2::new(vec![
            Seg2::line((1.1, 0.0)),
            Seg2::line((1.1 + shear, 0.12)),
            Seg2::line((shear, 0.12)),
            Seg2::line((0.0, 0.0)),
        ])
        .unwrap(),
    )
    .unwrap();
    let (rafter, rafter_regions) = extrusion(rafter_profile, 0.07);
    vec![
        Timber {
            key: "post",
            recipe: post,
            regions: post_regions,
            in_log: Affine3A::from_translation(Vec3::new(0.0, 0.0, 0.3)),
            placement: Placement3::IDENTITY,
        },
        Timber {
            key: "beam",
            recipe: beam,
            regions: beam_regions,
            in_log: Affine3A::from_translation(Vec3::new(0.0, 0.12, 0.9)),
            // Lying along x on top of the post.
            placement: Placement3::euler_extrinsic_xyz_then_translate(
                0.0,
                quarter,
                0.0,
                [-0.8, 0.0, 1.3],
            ),
        },
        Timber {
            key: "rafter",
            recipe: rafter,
            regions: rafter_regions,
            // The profile's x (the rafter's length) runs along the trunk,
            // 9 cm off the pith.
            in_log: Affine3A::from_rotation_translation(
                glam::Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
                Vec3::new(0.09, 0.02, 0.1),
            ),
            // Standing on its long edge, pitched up 35 degrees toward the
            // post so both end cuts are plumb.
            placement: Placement3::euler_extrinsic_xyz_then_translate(
                quarter,
                -35.0_f64.to_radians(),
                0.0,
                [-1.3, -0.35, 0.0],
            ),
        },
    ]
}

/// Baked textures, encoded as PNG, and the glTF materials that place them.
#[derive(Default)]
struct Baked {
    materials: BTreeMap<String, Value>,
    images: Vec<Vec<u8>>,
}

impl MaterialResolver for Baked {
    fn resolve(&self, key: &str) -> Option<Value> {
        self.materials.get(key).cloned()
    }

    fn resolve_texture(&self, index: u32) -> Option<Texture<'_>> {
        self.images
            .get(usize::try_from(index).ok()?)
            .map(|image| Texture {
                image,
                mime_type: "image/png",
                sampler: Some(json!({
                    "wrapS": 33071, "wrapT": 33071,
                    "magFilter": 9729, "minFilter": 9987
                })),
            })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os().nth(1).map_or_else(
        || repository().join(".local/gallery/timber-bake"),
        PathBuf::from,
    );
    std::fs::create_dir_all(&out)?;

    let timbers = timbers();
    let mut assembly = Assembly::new();
    let mut parts: Vec<PartId> = Vec::new();
    for timber in &timbers {
        let part = assembly.add_recipe_part(timber.key, timber.recipe.clone())?;
        for region in 0..timber.regions {
            let slot = format!("r{region}");
            assembly.bind_region_slot(part, region, &slot)?;
            assembly.set_part_material(part, &slot, &format!("{}/{slot}", timber.key))?;
        }
        assembly.add_instance(None, timber.key, part, timber.placement)?;
        parts.push(part);
    }
    let compiled = PartCompiler::new().compile_parts(&assembly, &CompilePolicy::default())?;

    let wood = oak::wood_channels()?;
    let mut baked = Baked::default();
    for (timber, part) in timbers.iter().zip(&parts) {
        let body = &compiled.part(*part).expect("compiled part").bodies[0];
        for range in &body.regions {
            let start = range.start as usize;
            let indices = &body.tri.indices[start..start + range.count as usize];
            let params = BakeParams::new(TEXELS_PER_METER).with_placement(timber.in_log);
            let bake = SurfaceBake::new(&body.tri, indices, &params)?;
            let mut channels = [
                vec![0.0; bake.samples().len()],
                vec![0.0; bake.samples().len()],
                vec![0.0; bake.samples().len()],
            ];
            for (program, values) in wood.iter().zip(&mut channels) {
                program.eval_chart_anisotropic(bake.samples(), values, 8);
            }
            let colors: Vec<[f32; 3]> = (0..bake.samples().len())
                .map(|i| [channels[0][i], channels[1][i], channels[2][i]])
                .collect();
            let texels = bake.scatter(&colors, [0.0; 3]);
            let png = encode_srgb(bake.width(), bake.height(), &texels)?;
            let name = format!("{}-r{}", timber.key, range.region);
            std::fs::write(out.join(format!("{name}.png")), &png)?;
            let stats = bake.stats();
            println!(
                "{name}: {} x {} texels, {} covered, {} dilated, {} overlapping",
                bake.width(),
                bake.height(),
                stats.covered_texels,
                stats.dilated_texels,
                stats.overlapping_texels
            );
            let transform = bake.texture_transform();
            let index = baked.images.len();
            baked.images.push(png);
            baked.materials.insert(
                format!("{}/r{}", timber.key, range.region),
                json!({
                    "name": name,
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {
                            "index": index,
                            "extensions": { "KHR_texture_transform": {
                                "offset": transform.offset,
                                "scale": transform.scale,
                            }},
                        },
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.62,
                    },
                }),
            );
        }
    }

    let glb = export_glb_with_materials(
        &assembly,
        &compiled,
        &baked,
        GltfExportOptions::z_up_to_y_up(),
    )?;
    let path = out.join("timbers.glb");
    std::fs::write(&path, glb.bytes)?;
    println!("{}", path.display());
    Ok(())
}

/// Encodes linear colors as an sRGB PNG, row 0 at the top.
fn encode_srgb(
    width: u32,
    height: u32,
    colors: &[[f32; 3]],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a value in [0, 255.5) rounds into a byte"
        )]
        let byte = (s * 255.0 + 0.5) as u8;
        byte
    };
    let pixels: Vec<u8> = colors.iter().flat_map(|c| c.map(encode)).collect();
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(BufWriter::new(&mut bytes), width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        encoder.write_header()?.write_image_data(&pixels)?;
    }
    Ok(bytes)
}

/// The repository root: this crate's manifest sits two levels below it.
fn repository() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("examples/timber_bake sits two levels below the repository")
}
