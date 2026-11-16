// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Small material operations exposed as module steps.

use alloc::vec;
use alloc::vec::Vec;

use dapple_field::PortType;
use dapple_material::module::{
    Args, Context, InputDecl, InputKind, Interface, Module, ModuleError, ModuleId, Output,
    OutputDecl, OutputKind, Outputs,
};
use dapple_material::ops::{self, Detail, Layer, Transition};
use dapple_material::{Aux, Channel, Material};

use super::{fail, scalar_of};

fn input(name: &'static str, kind: InputKind, doc: &'static str) -> InputDecl {
    InputDecl {
        name: name.into(),
        kind,
        required: true,
        doc: doc.into(),
    }
}

fn interface(name: &'static str, doc: &'static str, inputs: Vec<InputDecl>) -> Interface {
    Interface {
        id: ModuleId::new(name, 1),
        doc: doc.into(),
        params: vec![],
        inputs,
        outputs: vec![OutputDecl {
            name: "material".into(),
            kind: OutputKind::Material,
            doc: "the resulting material".into(),
        }],
    }
}

fn output(m: Material) -> Outputs {
    Outputs::new().with("material", Output::Material(m))
}

/// Adds a height map in meters to the base layer with [`ops::apply_detail`].
/// Existing geometry normals follow the added height's slope; all other
/// detail fields are the base layer's identity.
#[derive(Copy, Clone, Debug, Default)]
pub struct HeightDetail;

impl Module for HeightDetail {
    fn interface(&self) -> Interface {
        interface(
            "dapple_library.height_detail",
            "add geometric height detail to the base layer",
            vec![
                input("base", InputKind::Material, "the material to perturb"),
                input(
                    "height",
                    InputKind::Map(PortType::Scalar),
                    "height to add in meters",
                ),
            ],
        )
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let height = scalar_of(args.map("height").ok_or_else(|| fail("height"))?)?;
        Ok(output(cx.record(ops::apply_detail(
            base,
            &Detail {
                height: Some(height),
                ..Detail::identity(Layer::Base)
            },
        )?)))
    }
}

/// Selects between two materials with [`Transition::Mask`]: `mask` 0
/// chooses `base`, 1 chooses `other`, and intermediate values mix them.
#[derive(Copy, Clone, Debug, Default)]
pub struct SelectByMask;

impl Module for SelectByMask {
    fn interface(&self) -> Interface {
        interface(
            "dapple_library.select_by_mask",
            "select materials by a mask",
            vec![
                input("base", InputKind::Material, "the material where mask is 0"),
                input("other", InputKind::Material, "the material where mask is 1"),
                input(
                    "mask",
                    InputKind::Map(PortType::Mask),
                    "the other's share of each texel",
                ),
            ],
        )
    }

    fn build(&self, cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let base = args.material("base").ok_or_else(|| fail("base"))?;
        let other = args.material("other").ok_or_else(|| fail("other"))?;
        let mask = scalar_of(args.map("mask").ok_or_else(|| fail("mask"))?)?;
        Ok(output(cx.record(ops::select(
            base,
            other,
            &mask,
            Transition::Mask,
        )?)))
    }
}

/// Sets [`Aux::Region`] from an ID map without changing other channels.
/// Region labels identify units, independently of surface-material IDs.
#[derive(Copy, Clone, Debug, Default)]
pub struct SetRegion;

impl Module for SetRegion {
    fn interface(&self) -> Interface {
        interface(
            "dapple_library.set_region",
            "assign region labels to a material",
            vec![
                input("base", InputKind::Material, "the material to label"),
                input(
                    "region",
                    InputKind::Map(PortType::Id),
                    "the region label at each texel",
                ),
            ],
        )
    }

    fn build(&self, _cx: &mut Context<'_>, args: &Args) -> Result<Outputs, ModuleError> {
        let mut m = args.material("base").ok_or_else(|| fail("base"))?.clone();
        m.set_aux(
            Aux::Region,
            Channel::Map(args.map("region").ok_or_else(|| fail("region"))?.clone()),
        )?;
        Ok(output(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dapple_field::{Edge, Value};
    use dapple_material::module::Bind;
    use dapple_material::resource::NoResources;
    use dapple_material::{Grid, Param};
    use glam::Vec2;

    #[test]
    fn steps_preserve_the_operations_channels_and_reports() {
        let grid = Grid {
            width: 8,
            height: 8,
            origin: Vec2::ZERO,
            texel: Vec2::splat(0.125),
            edge: Edge::Wrap,
        };
        let base = Material::new(grid);
        let height = grid
            .raster(
                (0..grid.len())
                    .map(|i| if i % 8 < 4 { 0.002 } else { 0.0 })
                    .collect(),
            )
            .unwrap();
        let height_map = super::super::scalar_map(grid, &height).unwrap();
        let mut cx = Context::new(grid, &NoResources);
        let raised = cx
            .instantiate(
                &HeightDetail,
                "detail",
                Bind::new()
                    .material("base", base.clone())
                    .input("height", dapple_material::module::Input::Map(height_map)),
            )
            .unwrap()
            .take_material("material")
            .unwrap();
        let (expected, report) = ops::apply_detail(
            &base,
            &Detail {
                height: Some(height),
                ..Detail::identity(Layer::Base)
            },
        )
        .unwrap();
        assert_eq!(raised, expected);
        assert_eq!(
            cx.diagnostics().reports_at("detail").collect::<Vec<_>>(),
            vec![&report]
        );
        let mut other = raised;
        other
            .set_param(Param::CoatWeight, Channel::Constant(Value::Scalar(1.0)))
            .unwrap();
        let mask = grid
            .raster((0..grid.len()).map(|i| (i % 3) as f32 * 0.5).collect())
            .unwrap();
        let selected = cx
            .instantiate(
                &SelectByMask,
                "select",
                Bind::new()
                    .material("base", base.clone())
                    .material("other", other.clone())
                    .input(
                        "mask",
                        dapple_material::module::Input::Map(
                            super::super::mask_map(grid, &mask).unwrap(),
                        ),
                    ),
            )
            .unwrap()
            .take_material("material")
            .unwrap();
        let (expected, report) = ops::select(&base, &other, &mask, Transition::Mask).unwrap();
        assert_eq!(selected, expected);
        assert_eq!(
            cx.diagnostics().reports_at("select").collect::<Vec<_>>(),
            vec![&report]
        );
        let labels = grid
            .typed(
                PortType::Id,
                (0..grid.len()).map(|i| Value::Id(u32::try_from(i % 5).unwrap())),
            )
            .unwrap();
        let labeled = cx
            .instantiate(
                &SetRegion,
                "region",
                Bind::new().material("base", selected.clone()).input(
                    "region",
                    dapple_material::module::Input::Map(labels.clone()),
                ),
            )
            .unwrap()
            .take_material("material")
            .unwrap();
        let mut expected = selected;
        expected.set_aux(Aux::Region, Channel::Map(labels)).unwrap();
        assert_eq!(labeled, expected);
    }
}
