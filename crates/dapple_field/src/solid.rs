// Copyright 2026 the Dapple Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Solid (3D) fields: lattice noise, fractals, and cellular noise over
//! [`Domain3`], with the same contracts as their planar counterparts.
//!
//! Solid fields describe materials that exist through a volume, such as wood
//! grain or stone, and reach textures through a slice
//! ([`Op::Slice`](crate::program::Op::Slice)) or through chart evaluation at
//! surface points ([`SolidProgram::eval_chart`](crate::program::SolidProgram::eval_chart)).

use alloc::vec::Vec;

use glam::Vec3;

use crate::domain::{Domain3, DomainError, Footprint, Lattice3};
use crate::fractal::{FractalKind, FractalParams, MAX_OCTAVES};
use crate::hash::{hash, key, mix, unit_f32};
use crate::lanes;
use crate::noise::Basis;

/// Purpose tag for 3D lattice-corner hashes.
const LATTICE_TAG: u64 = 0x6c61_7474_6963_6533; // "lattice3"
/// Purpose tag for per-octave seeds.
const OCTAVE_TAG: u64 = 0x6f63_7461_7665; // "octave"
/// Purpose tag for per-octave lattice offsets.
const OFFSET_TAG: u64 = 0x6f66_6673_6574; // "offset"
/// Purpose tag for 3D feature-point hashes.
const FEATURE_TAG: u64 = 0x6665_6174_7572_6533; // "feature3"
/// Purpose tag for per-cell values.
const VALUE_TAG: u64 = 0x0076_616c_7565; // "value"

/// A point-evaluable scalar field over a [`Domain3`].
///
/// The solid counterpart of [`ScalarField`](crate::ScalarField): `eval` is
/// a pure function of its arguments, bit-identical on every platform, and
/// `footprint` is the side of the cube the value stands for.
pub trait SolidField {
    /// The domain this field is defined over.
    fn domain(&self) -> Domain3;

    /// Evaluates the field at `p`, band-limited to `footprint`.
    fn eval(&self, p: Vec3, footprint: Footprint) -> f32;

    /// The value at `p` and its gradient in domain units.
    ///
    /// The value must equal [`Self::eval`]'s, bit for bit. The default
    /// estimates the gradient by [`central_difference3`].
    fn eval_gradient(&self, p: Vec3, footprint: Footprint) -> (f32, Vec3) {
        (
            self.eval(p, footprint),
            central_difference3(self, p, footprint),
        )
    }
}

/// The gradient of `field` at `p` by central differences, stepping as
/// [`central_difference`](crate::central_difference) does.
pub fn central_difference3<F: SolidField + ?Sized>(
    field: &F,
    p: Vec3,
    footprint: Footprint,
) -> Vec3 {
    let h = (footprint.width() * 0.5).max(1e-4 * p.abs().max_element().max(1.0));
    let at = |q: Vec3| field.eval(q, footprint);
    let axis = |e: Vec3| (at(p + e * h) - at(p - e * h)) / (2.0 * h);
    Vec3::new(axis(Vec3::X), axis(Vec3::Y), axis(Vec3::Z))
}

impl<F: SolidField + ?Sized> SolidField for &F {
    fn domain(&self) -> Domain3 {
        (**self).domain()
    }

    fn eval(&self, p: Vec3, footprint: Footprint) -> f32 {
        (**self).eval(p, footprint)
    }

    fn eval_gradient(&self, p: Vec3, footprint: Footprint) -> (f32, Vec3) {
        (**self).eval_gradient(p, footprint)
    }
}

/// 256 unit gradients on a Fibonacci sphere, `z = 1 − (2k + 1)/256`, with
/// the azimuth offset by half a radian so none lies on an axis. Literals,
/// so every platform uses identical bits.
#[rustfmt::skip]
const GRADIENTS: [[f32; 3]; 256] = [
    [0.07749228, 0.04233423, 0.99609375],
    [-0.14820953, 0.03652541, 0.98828125],
    [0.109019496, -0.16369416, 0.98046875],
    [0.035646316, 0.22949763, 0.97265625],
    [-0.20517634, -0.16425349, 0.96484375],
    [0.2893428, -0.01928554, 0.95703125],
    [-0.21734132, 0.22747861, 0.94921875],
    [0.007076522, -0.33720052, 0.94140625],
    [0.23645353, 0.2692442, 0.93359375],
    [-0.37583566, -0.040946737, 0.92578125],
    [0.31977823, -0.23468117, 0.91796875],
    [-0.08070065, 0.40632868, 0.91015625],
    [-0.22365765, -0.36844674, 0.90234375],
    [0.42915004, 0.12507626, 0.89453125],
    [-0.41465607, 0.2044265, 0.88671875],
    [0.17299232, -0.44451937, 0.87890625],
    [0.17782271, 0.45779335, 0.87109375],
    [-0.45255598, -0.22346938, 0.86328125],
    [0.49726117, -0.14458403, 0.85546875],
    [-0.27559236, 0.45335168, 0.84765625],
    [-0.10540642, -0.532496, 0.83984375],
    [0.4470089, 0.32849205, 0.83203125],
    [-0.56297934, 0.06097323, 0.82421875],
    [0.38133678, -0.4336624, 0.81640625],
    [0.011970436, 0.5882456, 0.80859375],
    [-0.41349134, -0.43332934, 0.80078125],
    [0.6078875, 0.040906087, 0.79296875],
    [-0.48370725, 0.38672596, 0.78515625],
    [0.096948, -0.62156075, 0.77734375],
    [0.35365108, 0.53174484, 0.76953125],
    [-0.6289869, -0.15543498, 0.76171875],
    [0.5767563, -0.31460696, 0.75390625],
    [-0.21563722, 0.6299561, 0.74609375],
    [-0.26998836, -0.6180996, 0.73828125],
    [0.62432885, 0.2768192, 0.73046875],
    [-0.6551802, 0.22024262, 0.72265625],
    [0.3382444, -0.6120369, 0.71484375],
    [0.16586651, 0.68745553, 0.70703125],
    [-0.5930834, -0.39918074, 0.69921875],
    [0.7144384, -0.107402235, 0.69140625],
    [-0.45890605, 0.5675428, 0.68359375],
    [-0.045432825, -0.7357007, 0.67578125],
    [0.5355597, 0.51671416, 0.66796875],
    [-0.75087714, -0.019422997, 0.66015625],
    [0.5719209, -0.49734712, 0.65234375],
    [-0.086516514, 0.7596673, 0.64453125],
    [-0.4531841, -0.6238697, 0.63671875],
    [0.7618383, 0.15517502, 0.62890625],
    [-0.67193794, 0.40341288, 0.62109375],
    [0.22470808, -0.75722677, 0.61328125],
    [0.34843537, 0.715542, 0.60546875],
    [-0.74573946, -0.29441416, 0.59765625],
    [0.7541429, -0.28870904, 0.58984375],
    [-0.36358714, 0.7273541, 0.58203125],
    [-0.22474235, -0.7872507, 0.57421875],
    [0.70211947, 0.4315231, 0.56640625],
    [-0.81442976, 0.15708984, 0.55859375],
    [0.49752706, -0.67015433, 0.55078125],
    [0.08634651, 0.8353019, 0.54296875],
    [-0.63164634, -0.5609195, 0.53515625],
    [0.84955037, -0.013142132, 0.52734375],
    [-0.621043, 0.5868499, 0.51953125],
    [0.061864883, -0.8569228, 0.51171875],
    [0.53608406, 0.6772683, 0.50390625],
    [-0.85723317, -0.1379939, 0.49609375],
    [0.7290006, -0.47972855, 0.48828125],
    [-0.21454857, 0.85036385, 0.48046875],
    [-0.41822094, -0.7756851, 0.47265625],
    [0.83626664, 0.2908236, 0.46484375],
    [-0.81681204, 0.35205188, 0.45703125],
    [0.3661115, -0.8149631, 0.44921875],
    [0.28176057, 0.8519222, 0.44140625],
    [-0.7865445, -0.43970922, 0.43359375],
    [0.88061094, -0.20792963, 0.42578125],
    [-0.5109252, 0.7511708, 0.41796875],
    [-0.13117947, -0.90253186, 0.41015625],
    [0.70906943, 0.5790855, 0.40234375],
    [-0.9174008, 0.052162517, 0.39453125],
    [0.6435409, -0.66053283, 0.38671875],
    [-0.02844303, 0.92499787, 0.37890625],
    [-0.6059162, -0.70367247, 0.37109375],
    [0.92517024, 0.10993959, 0.36328125],
    [-0.75889766, 0.54563385, 0.35546875],
    [0.19161676, -0.9178334, 0.34765625],
    [0.48015538, 0.8086761, 0.33984375],
    [-0.9029719, -0.27275804, 0.33203125],
    [0.8525144, -0.41000167, 0.32421875],
    [-0.35264745, 0.88064003, 0.31640625],
    [-0.3357397, -0.8899712, 0.30859375],
    [0.8509609, 0.43057653, 0.30078125],
    [-0.92066115, 0.25797746, 0.29296875],
    [0.5058506, -0.81412596, 0.28515625],
    [0.17735837, 0.9442587, 0.27734375],
    [-0.77039284, -0.5777956, 0.26953125],
    [0.96050125, -0.09455524, 0.26171875],
    [-0.64576435, 0.7200833, 0.25390625],
    [-0.010264109, -0.9691917, 0.24609375],
    [0.6635805, 0.70914245, 0.23828125],
    [-0.97020036, -0.07480218, 0.23046875],
    [0.76735413, -0.6013251, 0.22265625],
    [-0.15992059, 0.9634664, 0.21484375],
    [-0.5338116, -0.8198678, 0.20703125],
    [0.94899833, 0.24436456, 0.19921875],
    [-0.8662008, 0.46158403, 0.19140625],
    [0.3274107, -0.9268741, 0.18359375],
    [0.38523042, 0.9059241, 0.17578125],
    [-0.8972405, -0.40834543, 0.16796875],
    [0.9386662, -0.30537802, 0.16015625],
    [-0.48647168, 0.860312, 0.15234375],
    [-0.2226872, -0.96411675, 0.14453125],
    [0.81636864, 0.5611152, 0.13671875],
    [-0.9820294, 0.13784571, 0.12890625],
    [0.6316308, -0.76575375, 0.12109375],
    [0.051562194, 0.9922241, 0.11328125],
    [-0.70887095, -0.6974083, 0.10546875],
    [0.994589, 0.03544016, 0.09765625],
    [-0.75787824, 0.64618003, 0.08984375],
    [0.12243028, -0.98908126, 0.08203125],
    [0.5781935, 0.812517, 0.07421875],
    [-0.97572774, -0.20867585, 0.06640625],
    [0.86085165, -0.5054712, 0.05859375],
    [-0.29345003, 0.9546247, 0.05078125],
    [-0.4286161, -0.9024644, 0.04296875],
    [0.92593706, 0.37603804, 0.03515625],
    [-0.93699616, 0.34826785, 0.02734375],
    [0.45574376, -0.8898967, 0.01953125],
    [0.2650977, 0.96415037, 0.01171875],
    [-0.8468007, -0.531896, 0.00390625],
    [0.98369503, -0.17980209, -0.00390625],
    [-0.6038547, 0.7970083, -0.01171875],
    [-0.09309632, -0.9954655, -0.01953125],
    [0.7409379, 0.6710167, -0.02734375],
    [-0.9993655, 0.0057081264, -0.03515625],
    [0.7328212, -0.67906314, -0.04296875],
    [-0.08162906, 0.99536824, -0.05078125],
    [-0.6119086, -0.7887551, -0.05859375],
    [0.9835165, 0.1681827, -0.06640625],
    [-0.8383573, 0.54004496, -0.07421875],
    [0.25322792, -0.96392244, -0.08203125],
    [0.46408388, 0.88122314, -0.08984375],
    [-0.93676627, -0.33605397, -0.09765625],
    [0.917008, -0.38467222, -0.10546875],
    [-0.41597086, 0.90229464, -0.11328125],
    [-0.30248636, -0.9454302, -0.12109375],
    [0.86081856, 0.49231556, -0.12890625],
    [-0.966274, 0.21822578, -0.13671875],
    [0.564458, -0.8127102, -0.14453125],
    [0.13260676, 0.97939104, -0.15234375],
    [-0.75839967, -0.63180685, -0.16015625],
    [0.9847018, -0.046355832, -0.16796875],
    [-0.6938149, 0.6983709, -0.17578125],
    [0.039796975, -0.9821963, -0.18359375],
    [0.633157, 0.7499839, -0.19140625],
    [-0.971934, -0.12512474, -0.19921875],
    [0.79986936, -0.56333566, -0.20703125],
    [-0.20891038, 0.9540433, -0.21484375],
    [-0.48952353, -0.84308416, -0.22265625],
    [0.9287201, 0.29045317, -0.23046875],
    [-0.8793023, 0.41237056, -0.23828125],
    [0.36907527, -0.89622617, -0.24609375],
    [0.3325539, 0.90826184, -0.25390625],
    [-0.85688615, -0.44412777, -0.26171875],
    [0.9297669, -0.25077173, -0.26953125],
    [-0.5149968, 0.81108487, -0.27734375],
    [-0.16773662, -0.94368976, -0.28515625],
    [0.7592637, 0.5811092, -0.29296875],
    [-0.9499717, 0.084169045, -0.30078125],
    [0.6419375, -0.70191604, -0.30859375],
    [0.0007905753, 0.9486235, -0.31640625],
    [-0.6395829, -0.69700503, -0.32421875],
    [0.93972504, 0.081682794, -0.33203125],
    [-0.74589, 0.57284755, -0.33984375],
    [0.16254732, -0.9234249, -0.34765625],
    [0.5023304, 0.7882297, -0.35546875],
    [-0.8999382, -0.2411181, -0.36328125],
    [0.82372355, -0.42868277, -0.37109375],
    [-0.31673542, 0.8695451, -0.37890625],
    [-0.35258073, -0.85213584, -0.38671875],
    [0.83258766, 0.38877094, -0.39453125],
    [-0.8732978, 0.27471888, -0.40234375],
    [0.4566336, -0.7894667, -0.41015625],
    [0.19580369, 0.8871094, -0.41796875],
    [-0.7406377, -0.5197751, -0.42578125],
    [0.8935398, -0.11654678, -0.43359375],
    [-0.577695, 0.6866069, -0.44140625],
    [-0.03765817, -0.8926278, -0.44921875],
    [0.6279259, 0.6299457, -0.45703125],
    [-0.88448143, -0.040160466, -0.46484375],
    [0.6761362, -0.5651866, -0.47265625],
    [-0.11622248, 0.86927676, -0.48046875],
    [-0.49901578, -0.71593624, -0.48828125],
    [0.84725624, 0.18986283, -0.49609375],
    [-0.7490791, 0.43006855, -0.50390625],
    [0.26044443, -0.81872624, -0.51171875],
    [0.3590225, 0.7753645, -0.51953125],
    [-0.78405434, -0.32736427, -0.52734375],
    [0.7946602, -0.28657097, -0.53515625],
    [-0.39005926, 0.74366575, -0.54296875],
    [-0.21341655, -0.8069036, -0.55078125],
    [0.6980391, 0.44801164, -0.55859375],
    [-0.81210214, 0.14026418, -0.56640625],
    [0.50075406, -0.6477023, -0.57421875],
    [0.06781451, 0.8103338, -0.58203125],
    [-0.59322697, -0.5478742, -0.58984375],
    [0.8017459, 0.0032430035, -0.59765625],
    [-0.5890188, 0.5352237, -0.60546875],
    [0.072236605, -0.7865545, -0.61328125],
    [0.47433588, 0.62389743, -0.62109375],
    [-0.7650421, -0.13851883, -0.62890625],
    [0.65228516, -0.41123387, -0.63671875],
    [-0.20147271, 0.73755556, -0.64453125],
    [-0.3466086, -0.6740253, -0.65234375],
    [0.7045028, 0.2605178, -0.66015625],
    [-0.689031, 0.28116545, -0.66796875],
    [0.31511572, -0.6663496, -0.67578125],
    [0.21561745, 0.69728667, -0.68359375],
    [-0.62361544, -0.36477548, -0.69140625],
    [0.6988483, -0.15067905, -0.69921875],
    [-0.40905806, 0.57686937, -0.70703125],
    [-0.087059684, -0.69384366, -0.71484375],
    [0.5267251, 0.44758087, -0.72265625],
    [-0.68247145, 0.025457533, -0.73046875],
    [0.48002103, -0.47383606, -0.73828125],
    [-0.033446282, 0.6650003, -0.74609375],
    [-0.41889063, -0.5061186, -0.75390625],
    [0.6417669, 0.08899316, -0.76171875],
    [-0.52567863, 0.36260667, -0.76953125],
    [0.14055185, -0.6131736, -0.77734375],
    [0.305727, 0.5385728, -0.78515625],
    [-0.57968605, -0.18752243, -0.79296875],
    [0.5447395, -0.24901465, -0.80078125],
    [-0.22933926, 0.5418299, -0.80859375],
    [-0.19324917, -0.54418343, -0.81640625],
    [0.5001879, 0.26547232, -0.82421875],
    [-0.5369736, 0.13922405, -0.83203125],
    [0.29542643, -0.4553962, -0.83984375],
    [0.08774609, 0.52323943, -0.84765625],
    [-0.4081418, -0.31873733, -0.85546875],
    [0.50316435, -0.03963759, -0.86328125],
    [-0.33496276, 0.35915962, -0.87109375],
    [0.004256262, -0.47697556, -0.87890625],
    [0.3092312, 0.34366545, -0.88671875],
    [-0.44492742, -0.04305161, -0.89453125],
    [0.34438196, -0.25918493, -0.90234375],
    [-0.07579465, 0.40727237, -0.91015625],
    [-0.2098998, -0.33656418, -0.91796875],
    [0.36420757, 0.101399794, -0.92578125],
    [-0.31946334, 0.16231418, -0.93359375],
    [0.11852059, -0.31576437, -0.94140625],
    [0.11744434, 0.2918743, -0.94921875],
    [-0.261542, -0.12524764, -0.95703125],
    [0.25146928, -0.07641819, -0.96484375],
    [-0.11824006, 0.19989774, -0.97265625],
    [-0.040514234, -0.19245681, -0.98046875],
    [0.124083936, 0.08890077, -0.98828125],
    [-0.08766757, 0.010565905, -0.99609375],
];

/// Squared radius of each corner's kernel, in cells, as in 2D.
const KERNEL_RADIUS_SQ: f32 = 2.25;

/// Weight of each corner's random value relative to its gradient term, as in 2D.
const VALUE_WEIGHT: f32 = 0.7;

/// Scales the sum into `[-1, 1]`: the largest `Σ K(r)·(r + 0.7)` over the
/// 4 × 4 × 4 corners around any point is below 2.475 (measured on a 41³ grid
/// of the cell, which the kernel's smoothness bounds).
const GRADIENT_SCALE: f32 = 1.0 / 2.49;

/// Quintic fade `6t⁵ − 15t⁴ + 10t³`.
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Derivative of [`fade`].
#[inline]
fn fade_derivative(t: f32) -> f32 {
    let s = t * (1.0 - t);
    30.0 * s * s
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// One octave of 3D lattice noise and, when `GRADIENT`, its gradient in
/// domain units. The value's arithmetic does not depend on `GRADIENT`.
fn lattice_eval3<const GRADIENT: bool>(
    basis: Basis,
    lattice: Lattice3,
    seed: u64,
    offset: Vec3,
    p: Vec3,
) -> (f32, Vec3) {
    let at = lattice.locate(p);
    let mut cell = at.cell;
    let mut f = at.frac + offset;
    // Both terms are below 1, so each sum is below 2 and subtracting 1 is exact.
    for axis in 0..3 {
        if f[axis] >= 1.0 {
            f[axis] -= 1.0;
            cell[axis] += 1;
        }
    }
    eval_located3::<GRADIENT>(basis, lattice, seed, cell, f)
}

/// [`lattice_eval3`] once its point is located: at `f` in `cell`, the
/// lattice offset applied.
#[inline(always)]
fn eval_located3<const GRADIENT: bool>(
    basis: Basis,
    lattice: Lattice3,
    seed: u64,
    cell: [i64; 3],
    f: Vec3,
) -> (f32, Vec3) {
    // A corner's hash is `hash(seed, [LATTICE_TAG, x, y, z])` of its
    // wrapped cell. The hash folds its keys from the left, so every corner
    // shares the seed-and-tag prefix, every corner of a column its `x`
    // prefix and every corner of a column and row its `x, y` prefix: mixing
    // those once leaves one mix per corner instead of four, for the same
    // bits, and wraps each column, row and layer once rather than at every
    // corner.
    let prefix = mix(seed, LATTICE_TAG);
    let column = |dx: i64| mix(prefix, key(lattice.wrap_axis(0, cell[0] + dx)));
    let row = |dy: i64| key(lattice.wrap_axis(1, cell[1] + dy));
    let layer = |dz: i64| key(lattice.wrap_axis(2, cell[2] + dz));
    match basis {
        Basis::Value => {
            let (x0, x1, y0, y1) = (column(0), column(1), row(0), row(1));
            let [x0y0, x1y0, x0y1, x1y1] = [mix(x0, y0), mix(x1, y0), mix(x0, y1), mix(x1, y1)];
            let (z0, z1) = (layer(0), layer(1));
            let c = |xy: u64, z: u64| unit_f32(mix(xy, z)) * 2.0 - 1.0;
            let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
            let (c000, c100, c010, c110) = (c(x0y0, z0), c(x1y0, z0), c(x0y1, z0), c(x1y1, z0));
            let (c001, c101, c011, c111) = (c(x0y0, z1), c(x1y0, z1), c(x0y1, z1), c(x1y1, z1));
            let x00 = lerp(c000, c100, u);
            let x10 = lerp(c010, c110, u);
            let x01 = lerp(c001, c101, u);
            let x11 = lerp(c011, c111, u);
            let y0 = lerp(x00, x10, v);
            let y1 = lerp(x01, x11, v);
            let result = lerp(y0, y1, w);
            if !GRADIENT {
                return (result, Vec3::ZERO);
            }
            let (du, dv, dw) = (
                fade_derivative(f.x),
                fade_derivative(f.y),
                fade_derivative(f.z),
            );
            // ∂/∂u of the trilinear blend of the corner values.
            let dx0 = lerp(c100 - c000, c110 - c010, v);
            let dx1 = lerp(c101 - c001, c111 - c011, v);
            let dfx = lerp(dx0, dx1, w) * du;
            let dfy = lerp(x10 - x00, x11 - x01, w) * dv;
            let dfz = (y1 - y0) * dw;
            (result, Vec3::new(dfx, dfy, dfz) * lattice.frequency)
        }
        Basis::Gradient => {
            // The layers first: wrapped after the lines, they cost 3% on a
            // periodic lattice.
            let layers = gradient_layers3(lattice, cell[2]);
            let lines = gradient_lines3(lattice, prefix, [cell[0], cell[1]]);
            gradient_sum3::<GRADIENT>(lattice, &lines, &layers, f)
        }
    }
}

/// The `x, y` hash prefixes of the 16 lines of corners along `z` that
/// [`Basis::Gradient`] sums around a cell in column `x, y`, from `prefix`,
/// the seed and tag mixed: line `(i, j)`, `i - 1` and `j - 1` cells from
/// the column, is `lines[i + 4j]`.
///
/// A function of the column alone, so points in the same column share it.
#[inline(always)]
fn gradient_lines3(lattice: Lattice3, prefix: u64, [x, y]: [i64; 2]) -> [u64; 16] {
    let column = |dx: i64| mix(prefix, key(lattice.wrap_axis(0, x + dx)));
    let row = |dy: i64| key(lattice.wrap_axis(1, y + dy));
    // Written out rather than `array::map`, which is not always inlined.
    let columns = [column(-1), column(0), column(1), column(2)];
    let rows = [row(-1), row(0), row(1), row(2)];
    let mut lines = [0_u64; 16];
    for (j, row) in rows.into_iter().enumerate() {
        for (i, column) in columns.into_iter().enumerate() {
            lines[i + 4 * j] = mix(column, row);
        }
    }
    lines
}

/// The keys of the four layers of corners [`Basis::Gradient`] sums around
/// a cell in layer `z`, `k - 1` cells from it at `k`.
#[inline(always)]
fn gradient_layers3(lattice: Lattice3, z: i64) -> [u64; 4] {
    let layer = |dz: i64| key(lattice.wrap_axis(2, z + dz));
    [layer(-1), layer(0), layer(1), layer(2)]
}

/// [`Basis::Gradient`] noise, with its gradient when `GRADIENT`, at `f` in
/// a cell whose corners' hash prefixes are `lines` and `layers`.
#[inline(always)]
fn gradient_sum3<const GRADIENT: bool>(
    lattice: Lattice3,
    lines: &[u64; 16],
    layers: &[u64; 4],
    f: Vec3,
) -> (f32, Vec3) {
    // The point's offset from the corners on each axis, and its square.
    let offsets = |f: f32| [f - -1.0, f - 0.0, f - 1.0, f - 2.0];
    let (ox, oy, oz) = (offsets(f.x), offsets(f.y), offsets(f.z));
    let squares = |o: [f32; 4]| [o[0] * o[0], o[1] * o[1], o[2] * o[2], o[3] * o[3]];
    let (sx, sy, sz) = (squares(ox), squares(oy), squares(oz));
    // Corner `(i, j, k)` is bit `i + 4j + 16k`, so rising bits run along
    // rows within layers, the order the sums take. About 14 of the 64
    // corners fall inside the kernel, and which ones changes from point to
    // point: a branch per corner mispredicts. The inside corners are
    // marked without branching and visited by their bits, a loop whose
    // only unpredictable branch is its end. A NaN distance is not outside,
    // so its corner counts, as it did when this was a `continue`.
    let mut inside = 0_u64;
    for (k, z2) in sz.into_iter().enumerate() {
        for (j, y2) in sy.into_iter().enumerate() {
            for (i, x2) in sx.into_iter().enumerate() {
                let outside = x2 + y2 + z2 >= KERNEL_RADIUS_SQ;
                inside |= u64::from(!outside) << (i + 4 * j + 16 * k);
            }
        }
    }
    let mut sum = 0.0;
    let mut gradient = Vec3::ZERO;
    while inside != 0 {
        let index = inside.trailing_zeros() as usize;
        inside &= inside - 1;
        let (i, j, k) = (index & 3, (index >> 2) & 3, index >> 4);
        let d = Vec3::new(ox[i], oy[j], oz[k]);
        let r2 = sx[i] + sy[j] + sz[k];
        let h = mix(lines[index & 15], layers[k]);
        // The top byte picks the gradient; the next 24 bits give the
        // corner's value.
        let [gx, gy, gz] = GRADIENTS[(h >> 56) as usize];
        #[expect(
            clippy::cast_precision_loss,
            reason = "a 24-bit integer is exact in f32"
        )]
        let value = ((h >> 32) & 0xFF_FFFF) as f32 * (1.0 / 16_777_216.0) * 2.0 - 1.0;
        let t = 1.0 - r2 / KERNEL_RADIUS_SQ;
        let t2 = t * t;
        let term = gx * d.x + gy * d.y + gz * d.z + VALUE_WEIGHT * value;
        sum += t2 * t2 * term;
        if GRADIENT {
            // d/dd of t⁴·term: 4t³·(−2d/R²)·term + t⁴·g.
            let dk = -8.0 * t2 * t / KERNEL_RADIUS_SQ;
            gradient += d * (dk * term) + Vec3::new(gx, gy, gz) * (t2 * t2);
        }
    }
    (
        sum * GRADIENT_SCALE,
        gradient * GRADIENT_SCALE * lattice.frequency,
    )
}

/// One octave of 3D lattice noise at the points `(x[i], y[i], z[i])`, into
/// `out`: at each point the bits of [`lattice_eval3`]'s value there.
///
/// A lane kernel, run a chunk of [`lanes::WIDTH`] points at a time. The
/// chunk's points are located on the lattice together, in loops that
/// vectorize. [`Basis::Value`] then hashes the eight corners of every point
/// in a loop of its own, scalar since the hash's 64-bit multiplies have no
/// vector form, and blends them over the chunk, vectorized again.
/// [`Basis::Gradient`], whose corners inside the kernel vary from point to
/// point, sums each point's corners as [`lattice_eval3`] does, hashing the
/// prefixes of its 16 lines of corners only when a lane's column of cells
/// differs from the lane before's.
pub(crate) fn lattice_lanes3(
    basis: Basis,
    lattice: Lattice3,
    seed: u64,
    offset: Vec3,
    points: [&[f32]; 3],
    out: &mut [f32],
) {
    for (points, out) in lanes::chunks(points, out) {
        let n = out.len().min(lanes::WIDTH);
        let mut cells = [[0_i64; lanes::WIDTH]; 3];
        let mut fracs = [[0.0_f32; lanes::WIDTH]; 3];
        for (axis, (cells, fracs)) in cells.iter_mut().zip(&mut fracs).enumerate() {
            lanes::locate(
                points[axis],
                lattice.frequency[axis],
                offset[axis],
                &mut cells[..n],
                &mut fracs[..n],
            );
        }
        match basis {
            Basis::Value => value_lanes3(lattice, seed, &cells, &fracs, &mut out[..n]),
            Basis::Gradient => {
                // A cell's corners' hash prefixes are a function of the
                // cell alone, and neighboring points often share a cell (at
                // low frequencies, a run of them): a lane in the same
                // column as the lane before keeps its lines, and in the
                // same layer its layers. The first lane computes both.
                let prefix = mix(seed, LATTICE_TAG);
                let ([cx, cy, cz], [fx, fy, fz]) = (&cells, &fracs);
                let (mut column, mut layer) = ([cx[0], cy[0]], cz[0]);
                let mut lines = gradient_lines3(lattice, prefix, column);
                let mut layers = gradient_layers3(lattice, layer);
                for (i, out) in out[..n].iter_mut().enumerate() {
                    if [cx[i], cy[i]] != column {
                        column = [cx[i], cy[i]];
                        lines = gradient_lines3(lattice, prefix, column);
                    }
                    if cz[i] != layer {
                        layer = cz[i];
                        layers = gradient_layers3(lattice, layer);
                    }
                    let f = Vec3::new(fx[i], fy[i], fz[i]);
                    *out = gradient_sum3::<false>(lattice, &lines, &layers, f).0;
                }
            }
        }
    }
}

/// [`Basis::Value`] noise at the located points of a chunk, the first
/// `out.len()` of `cells` and `fracs`.
fn value_lanes3(
    lattice: Lattice3,
    seed: u64,
    cells: &[[i64; lanes::WIDTH]; 3],
    fracs: &[[f32; lanes::WIDTH]; 3],
    out: &mut [f32],
) {
    let n = out.len().min(lanes::WIDTH);
    // The corners' values, `c[i + 2j + 4k]` for corner `(i, j, k)`, with a
    // loop of its own for each kind of lattice, so that neither tests the
    // kind per point.
    let mut c = [[0.0_f32; lanes::WIDTH]; 8];
    let prefix = mix(seed, LATTICE_TAG);
    match lattice.wrap {
        None => value_corners3(prefix, cells, n, |_, index| index, &mut c),
        Some(period) => value_corners3(
            prefix,
            cells,
            n,
            |axis, index: i64| index.rem_euclid(period[axis]),
            &mut c,
        ),
    }
    let [fx, fy, fz] = fracs;
    for (i, out) in out[..n].iter_mut().enumerate() {
        let (u, v, w) = (fade(fx[i]), fade(fy[i]), fade(fz[i]));
        let x00 = lerp(c[0][i], c[1][i], u);
        let x10 = lerp(c[2][i], c[3][i], u);
        let x01 = lerp(c[4][i], c[5][i], u);
        let x11 = lerp(c[6][i], c[7][i], u);
        let y0 = lerp(x00, x10, v);
        let y1 = lerp(x01, x11, v);
        *out = lerp(y0, y1, w);
    }
}

/// The values of the eight corners of each of the first `n` cells, into
/// `out[i + 2j + 4k]` for corner `(i, j, k)`, hashed as [`lattice_eval3`]
/// hashes them from `prefix`, its seed and tag mixed, with `wrap(axis,
/// index)` wrapping a coordinate as the lattice does.
#[inline(always)]
fn value_corners3(
    prefix: u64,
    cells: &[[i64; lanes::WIDTH]; 3],
    n: usize,
    wrap: impl Fn(usize, i64) -> i64,
    out: &mut [[f32; lanes::WIDTH]; 8],
) {
    for i in 0..n.min(lanes::WIDTH) {
        let [x, y, z] = [cells[0][i], cells[1][i], cells[2][i]];
        let (x0, x1) = (
            mix(prefix, key(wrap(0, x))),
            mix(prefix, key(wrap(0, x + 1))),
        );
        let (y0, y1) = (key(wrap(1, y)), key(wrap(1, y + 1)));
        let (z0, z1) = (key(wrap(2, z)), key(wrap(2, z + 1)));
        let [x0y0, x1y0, x0y1, x1y1] = [mix(x0, y0), mix(x1, y0), mix(x0, y1), mix(x1, y1)];
        let value = |xy: u64, z: u64| unit_f32(mix(xy, z)) * 2.0 - 1.0;
        out[0][i] = value(x0y0, z0);
        out[1][i] = value(x1y0, z0);
        out[2][i] = value(x0y1, z0);
        out[3][i] = value(x1y1, z0);
        out[4][i] = value(x0y0, z1);
        out[5][i] = value(x1y0, z1);
        out[6][i] = value(x0y1, z1);
        out[7][i] = value(x1y1, z1);
    }
}

/// One octave of 3D lattice noise in about `[-1, 1]`, with mean zero.
///
/// The solid counterpart of [`Noise`](crate::Noise): the same bases, radial
/// kernels of 1.5 cells over the 4 × 4 × 4 surrounding corners for
/// [`Basis::Gradient`], a quintic trilinear blend for [`Basis::Value`], and
/// the same footprint fade. On a periodic domain each axis must fit a whole
/// number of cells into the period, and repeats are bit-identical.
///
/// ```
/// use dapple_field::{Basis, Domain3, Footprint, Noise3, SolidField};
/// use glam::Vec3;
///
/// let domain = Domain3::periodic(1, 1, 1).unwrap();
/// let noise = Noise3::new(Basis::Gradient, domain, Vec3::splat(4.0), 7)?;
/// let a = noise.eval(Vec3::new(0.0, 0.25, 0.5), Footprint::POINT);
/// let b = noise.eval(Vec3::new(1.0, 0.25, 1.5), Footprint::POINT);
/// assert_eq!(a.to_bits(), b.to_bits(), "the period wraps exactly");
/// # Ok::<(), dapple_field::DomainError>(())
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Noise3 {
    basis: Basis,
    domain: Domain3,
    lattice: Lattice3,
    seed: u64,
}

impl Noise3 {
    /// Builds noise of `basis` at `frequency` cells per unit.
    pub fn new(
        basis: Basis,
        domain: Domain3,
        frequency: Vec3,
        seed: u64,
    ) -> Result<Self, DomainError> {
        Ok(Self {
            basis,
            domain,
            lattice: Lattice3::new(domain, frequency)?,
            seed,
        })
    }

    /// The noise flavor.
    #[must_use]
    pub const fn basis(&self) -> Basis {
        self.basis
    }

    /// The noise at the points `(x[i], y[i], z[i])`, all with `footprint`,
    /// into `out`: each value the bits [`SolidField::eval`] gives at its
    /// point.
    pub(crate) fn eval_lanes(&self, points: [&[f32]; 3], footprint: Footprint, out: &mut [f32]) {
        let weight = footprint.band_weight(self.lattice.max_frequency());
        if weight == 0.0 {
            out.fill(0.0);
            return;
        }
        lattice_lanes3(self.basis, self.lattice, self.seed, Vec3::ZERO, points, out);
        for value in out {
            *value *= weight;
        }
    }
}

impl SolidField for Noise3 {
    fn domain(&self) -> Domain3 {
        self.domain
    }

    fn eval(&self, p: Vec3, footprint: Footprint) -> f32 {
        let weight = footprint.band_weight(self.lattice.max_frequency());
        if weight == 0.0 {
            return 0.0;
        }
        lattice_eval3::<false>(self.basis, self.lattice, self.seed, Vec3::ZERO, p).0 * weight
    }

    fn eval_gradient(&self, p: Vec3, footprint: Footprint) -> (f32, Vec3) {
        let weight = footprint.band_weight(self.lattice.max_frequency());
        if weight == 0.0 {
            return (0.0, Vec3::ZERO);
        }
        let (n, gradient) =
            lattice_eval3::<true>(self.basis, self.lattice, self.seed, Vec3::ZERO, p);
        (n * weight, gradient * weight)
    }
}

/// Mean of one ridged 3D octave, `E[(1 − |n|)²]`, per basis; measured over
/// 640k samples (8 seeds), like the planar means, and checked by a test.
const fn ridged_mean3(basis: Basis) -> f32 {
    match basis {
        Basis::Value => 0.500,
        Basis::Gradient => 0.753,
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Octave3 {
    lattice: Lattice3,
    seed: u64,
    offset: Vec3,
    amplitude: f32,
}

/// A fractal sum of 3D lattice-noise octaves, band-limited by footprint.
///
/// The solid counterpart of [`Fractal`](crate::Fractal), with the same octave
/// structure, keyed seeds and sub-cell lattice offsets, normalization, and
/// footprint fade to each octave's mean.
#[derive(Clone, Debug, PartialEq)]
pub struct Fractal3 {
    basis: Basis,
    kind: FractalKind,
    domain: Domain3,
    octaves: Vec<Octave3>,
    total_amplitude: f32,
}

impl Fractal3 {
    /// Builds a fractal whose first octave has `frequency` cells per unit.
    pub fn new(
        basis: Basis,
        domain: Domain3,
        frequency: Vec3,
        seed: u64,
        params: FractalParams,
    ) -> Result<Self, DomainError> {
        if params.octaves == 0 || params.octaves > MAX_OCTAVES {
            return Err(DomainError::InvalidParameter { name: "octaves" });
        }
        if params.lacunarity < 2 {
            return Err(DomainError::InvalidParameter { name: "lacunarity" });
        }
        if !(params.gain > 0.0 && params.gain <= 1.0) {
            return Err(DomainError::InvalidParameter { name: "gain" });
        }
        let mut lattice = Lattice3::new(domain, frequency)?;
        let mut amplitude = 1.0;
        let mut octaves = Vec::with_capacity(usize::from(params.octaves));
        for i in 0..params.octaves {
            if i > 0 {
                lattice = lattice.refined(params.lacunarity)?;
                amplitude *= params.gain;
            }
            let octave_seed = hash(seed, &[OCTAVE_TAG, u64::from(i)]);
            octaves.push(Octave3 {
                lattice,
                seed: octave_seed,
                offset: Vec3::new(
                    unit_f32(hash(octave_seed, &[OFFSET_TAG, 0])),
                    unit_f32(hash(octave_seed, &[OFFSET_TAG, 1])),
                    unit_f32(hash(octave_seed, &[OFFSET_TAG, 2])),
                ),
                amplitude,
            });
        }
        let total_amplitude = octaves.iter().map(|o| o.amplitude).sum();
        Ok(Self {
            basis,
            kind: params.kind,
            domain,
            octaves,
            total_amplitude,
        })
    }

    fn mean(&self) -> f32 {
        match self.kind {
            FractalKind::Fbm => 0.0,
            FractalKind::Ridged => ridged_mean3(self.basis),
        }
    }

    /// The fractal at the points `(x[i], y[i], z[i])`, all with
    /// `footprint`, into `out`: octave by octave over each chunk of
    /// [`lanes::WIDTH`] points, each octave's band weight taken once and its
    /// noise by the lane kernel, and each value the bits
    /// [`SolidField::eval`] gives at its point.
    pub(crate) fn eval_lanes(&self, [x, y, z]: [&[f32]; 3], footprint: Footprint, out: &mut [f32]) {
        let mean = self.mean();
        let mut noise = [0.0; lanes::WIDTH];
        for (points, out) in lanes::chunks([x, y, z], out) {
            let noise = &mut noise[..out.len()];
            out.fill(0.0);
            for octave in &self.octaves {
                let weight = footprint.band_weight(octave.lattice.max_frequency());
                if weight == 0.0 {
                    for sum in &mut *out {
                        *sum += mean * octave.amplitude;
                    }
                    continue;
                }
                let (lattice, seed, offset) = (octave.lattice, octave.seed, octave.offset);
                lattice_lanes3(self.basis, lattice, seed, offset, points, noise);
                let lanes = out.iter_mut().zip(&*noise);
                match self.kind {
                    FractalKind::Fbm => {
                        for (sum, &detail) in lanes {
                            *sum += (mean + (detail - mean) * weight) * octave.amplitude;
                        }
                    }
                    FractalKind::Ridged => {
                        for (sum, &n) in lanes {
                            let r = 1.0 - n.abs();
                            let detail = r * r;
                            *sum += (mean + (detail - mean) * weight) * octave.amplitude;
                        }
                    }
                }
            }
            for sum in out {
                *sum /= self.total_amplitude;
            }
        }
    }

    fn sum<const GRADIENT: bool>(&self, p: Vec3, footprint: Footprint) -> (f32, Vec3) {
        let mean = self.mean();
        let mut sum = 0.0;
        let mut gradient = Vec3::ZERO;
        for octave in &self.octaves {
            let weight = footprint.band_weight(octave.lattice.max_frequency());
            let value = if weight == 0.0 {
                mean
            } else {
                let (n, dn) = lattice_eval3::<GRADIENT>(
                    self.basis,
                    octave.lattice,
                    octave.seed,
                    octave.offset,
                    p,
                );
                let (detail, d_detail) = match self.kind {
                    FractalKind::Fbm => (n, dn),
                    FractalKind::Ridged => {
                        let r = 1.0 - n.abs();
                        (r * r, dn * (-2.0 * r * n.signum()))
                    }
                };
                if GRADIENT {
                    gradient += d_detail * (weight * octave.amplitude);
                }
                mean + (detail - mean) * weight
            };
            sum += value * octave.amplitude;
        }
        (sum / self.total_amplitude, gradient / self.total_amplitude)
    }
}

impl SolidField for Fractal3 {
    fn domain(&self) -> Domain3 {
        self.domain
    }

    fn eval(&self, p: Vec3, footprint: Footprint) -> f32 {
        self.sum::<false>(p, footprint).0
    }

    fn eval_gradient(&self, p: Vec3, footprint: Footprint) -> (f32, Vec3) {
        self.sum::<true>(p, footprint)
    }
}

/// 3D Worley noise with one jittered feature point per lattice cell.
///
/// The solid counterpart of [`Cellular`](crate::Cellular), selected as a
/// field with [`Self::output`]. Distances are in lattice-cell units. With
/// `jitter` in `[0, 1]` every point lies in its own cell, so the nearest two
/// points are within √4.25 < 2.1 cells of the query and every cell four or
/// more steps away is at least 3 away: searching the 7 × 7 × 7 block around
/// the query's cell is exact for `f1`, `f2`, and the border distance.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Cellular3 {
    domain: Domain3,
    lattice: Lattice3,
    jitter: f32,
    seed: u64,
}

/// One 3D cellular lookup; see [`CellSample`](crate::CellSample).
#[derive(Copy, Clone, Debug, PartialEq)]
struct CellSample3 {
    f1: f32,
    f2: f32,
    border: f32,
    value: f32,
}

impl Cellular3 {
    /// Builds cellular noise with `frequency` cells per unit.
    pub fn new(
        domain: Domain3,
        frequency: Vec3,
        jitter: f32,
        seed: u64,
    ) -> Result<Self, DomainError> {
        if !(0.0..=1.0).contains(&jitter) {
            return Err(DomainError::InvalidParameter { name: "jitter" });
        }
        Ok(Self {
            domain,
            lattice: Lattice3::new(domain, frequency)?,
            jitter,
            seed,
        })
    }

    /// Selects one quantity as a solid field, with its band-limiting mean
    /// computed as [`Cellular::output`](crate::Cellular::output) does, over a
    /// 16³ stratified sample of a 4 × 4 × 4-cell block.
    #[must_use]
    pub fn output(self, output: crate::CellOutput) -> CellularField3 {
        let mean = match output {
            crate::CellOutput::CellValue => 0.5,
            _ => {
                const SIDE: u32 = 16;
                const CELLS: f32 = 4.0;
                let mut sum = 0.0_f64;
                for k in 0..SIDE {
                    for j in 0..SIDE {
                        for i in 0..SIDE {
                            let cell = Vec3::new(
                                (i as f32 + 0.5) * CELLS / SIDE as f32,
                                (j as f32 + 0.5) * CELLS / SIDE as f32,
                                (k as f32 + 0.5) * CELLS / SIDE as f32,
                            );
                            let s = self.sample::<false>(cell / self.lattice.frequency, true).0;
                            sum += f64::from(select(output, &s));
                        }
                    }
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the mean is narrowed back to the field's f32"
                )]
                let mean = (sum / f64::from(SIDE * SIDE * SIDE)) as f32;
                mean
            }
        };
        CellularField3 {
            cellular: self,
            output,
            mean,
        }
    }

    /// Feature point of the cell `offset` steps from the query cell, whose
    /// id (its hash) is `id`, relative to the query cell's corner.
    fn feature(&self, id: u64, offset: [i8; 3]) -> Vec3 {
        let jitter = Vec3::new(
            unit_f32(hash(id, &[0])) - 0.5,
            unit_f32(hash(id, &[1])) - 0.5,
            unit_f32(hash(id, &[2])) - 0.5,
        ) * self.jitter;
        let corner = Vec3::new(
            f32::from(offset[0]),
            f32::from(offset[1]),
            f32::from(offset[2]),
        );
        corner + Vec3::splat(0.5) + jitter
    }

    /// The feature points and ids of the 7 × 7 × 7 cells around `cell`,
    /// rows within layers from the bottom, relative to `cell`'s corner.
    ///
    /// Inlined, so that the 8 KiB result is built in place rather than
    /// copied out.
    #[inline]
    fn neighborhood(&self, cell: [i64; 3]) -> [(Vec3, u64); 343] {
        // A cell's id is `hash(seed, [FEATURE_TAG, x, y, z])` of its wrapped
        // index. The hash folds its keys from the left, so the seed-and-tag
        // prefix is mixed once, each column's `x` once, each column and
        // row's `y` once, and each cell's `z` last: one mix per cell instead
        // of four for the same bits, and one wrap per column, row and layer
        // rather than three per cell.
        let prefix = mix(self.seed, FEATURE_TAG);
        let offsets = [-3, -2, -1, 0, 1, 2, 3_i8];
        let wrap =
            |axis: usize, d: i8| key(self.lattice.wrap_axis(axis, cell[axis] + i64::from(d)));
        let mut columns = [0_u64; 7];
        let mut rows = [0_u64; 7];
        let mut layers = [0_u64; 7];
        for (i, d) in offsets.into_iter().enumerate() {
            columns[i] = mix(prefix, wrap(0, d));
            rows[i] = wrap(1, d);
            layers[i] = wrap(2, d);
        }
        let mut lines = [[0_u64; 7]; 7];
        for (strip, row) in lines.iter_mut().zip(rows) {
            for (line, column) in strip.iter_mut().zip(columns) {
                *line = mix(column, row);
            }
        }
        let mut points = [(Vec3::ZERO, 0_u64); 343];
        for ((dz, layer), block) in offsets
            .into_iter()
            .zip(layers)
            .zip(points.as_chunks_mut::<49>().0)
        {
            for ((dy, strip), slots) in offsets
                .into_iter()
                .zip(&lines)
                .zip(block.as_chunks_mut::<7>().0)
            {
                for ((dx, &line), slot) in offsets.into_iter().zip(strip).zip(slots) {
                    let id = mix(line, layer);
                    *slot = (self.feature(id, [dx, dy, dz]), id);
                }
            }
        }
        points
    }

    /// The lookup at `p` and, when `GRADIENT`, the in-cell gradients of
    /// `f1`, `f2` and the border distance.
    ///
    /// The border distance, by far the costliest part, is searched for only
    /// when `border`; otherwise it and its gradient are left infinite and
    /// zero.
    fn sample<const GRADIENT: bool>(&self, p: Vec3, border: bool) -> (CellSample3, [Vec3; 3]) {
        let at = self.lattice.locate(p);
        search3::<GRADIENT>(&self.neighborhood(at.cell), at.frac, border)
    }
}

/// [`Cellular3::sample`] at `q` in a cell whose
/// [`Cellular3::neighborhood`] is `points`.
#[inline(always)]
fn search3<const GRADIENT: bool>(
    points: &[(Vec3, u64); 343],
    q: Vec3,
    border: bool,
) -> (CellSample3, [Vec3; 3]) {
    let mut nearest = (f32::INFINITY, 0);
    let mut second = (f32::INFINITY, 0);
    for (index, (point, _)) in points.iter().enumerate() {
        let d = (*point - q).length_squared();
        if d < nearest.0 {
            second = nearest;
            nearest = (d, index);
        } else if d < second.0 {
            second = (d, index);
        }
    }
    let (near_point, id) = points[nearest.1];
    let (border, border_axis) = if border {
        border_search3::<GRADIENT>(points, nearest.1, q)
    } else {
        (f32::INFINITY, Vec3::ZERO)
    };
    let sample = CellSample3 {
        f1: libm::sqrtf(nearest.0),
        f2: libm::sqrtf(second.0),
        border,
        value: unit_f32(hash(id, &[VALUE_TAG])),
    };
    if !GRADIENT {
        return (sample, [Vec3::ZERO; 3]);
    }
    let away = |point: Vec3, distance: f32| {
        if distance > 0.0 {
            (q - point) / distance
        } else {
            Vec3::ZERO
        }
    };
    (
        sample,
        [
            away(near_point, sample.f1),
            away(points[second.1].0, sample.f2),
            -border_axis,
        ],
    )
}

/// The distance from `q` to the nearest bisector between `points[nearest]`
/// and another of `points`, and when `GRADIENT` that bisector's unit normal
/// away from the nearest point.
#[inline(always)]
fn border_search3<const GRADIENT: bool>(
    points: &[(Vec3, u64); 343],
    nearest: usize,
    q: Vec3,
) -> (f32, Vec3) {
    let near_point = points[nearest].0;
    let mut border = f32::INFINITY;
    let mut border_axis = Vec3::ZERO;
    for (index, (point, _)) in points.iter().enumerate() {
        if index == nearest {
            continue;
        }
        let axis = *point - near_point;
        let length = axis.length();
        if length > 0.0 {
            let midpoint = (*point + near_point) * 0.5;
            let distance = (midpoint - q).dot(axis) / length;
            if GRADIENT && distance < border {
                border_axis = axis / length;
            }
            border = border.min(distance);
        }
    }
    (border, border_axis)
}

fn select(output: crate::CellOutput, s: &CellSample3) -> f32 {
    use crate::CellOutput;
    match output {
        CellOutput::F1 => s.f1,
        CellOutput::F2 => s.f2,
        CellOutput::F2MinusF1 => s.f2 - s.f1,
        CellOutput::Border => s.border,
        CellOutput::CellValue => s.value,
    }
}

/// One output of a [`Cellular3`] as a solid field, band-limited like
/// [`CellularField`](crate::CellularField) and with analytic gradients.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CellularField3 {
    cellular: Cellular3,
    output: crate::CellOutput,
    mean: f32,
}

impl CellularField3 {
    /// The value coarse footprints fade toward.
    #[must_use]
    pub const fn mean(&self) -> f32 {
        self.mean
    }
}

impl SolidField for CellularField3 {
    fn domain(&self) -> Domain3 {
        self.cellular.domain
    }

    fn eval(&self, p: Vec3, footprint: Footprint) -> f32 {
        let weight = footprint.band_weight(self.cellular.lattice.max_frequency());
        if weight == 0.0 {
            return self.mean;
        }
        let border = self.output == crate::CellOutput::Border;
        let value = select(self.output, &self.cellular.sample::<false>(p, border).0);
        if weight == 1.0 {
            value
        } else {
            self.mean + (value - self.mean) * weight
        }
    }

    fn eval_gradient(&self, p: Vec3, footprint: Footprint) -> (f32, Vec3) {
        use crate::CellOutput;
        let weight = footprint.band_weight(self.cellular.lattice.max_frequency());
        if weight == 0.0 {
            return (self.mean, Vec3::ZERO);
        }
        let border = self.output == CellOutput::Border;
        let (sample, [df1, df2, dborder]) = self.cellular.sample::<true>(p, border);
        let value = select(self.output, &sample);
        let in_cell = match self.output {
            CellOutput::F1 => df1,
            CellOutput::F2 => df2,
            CellOutput::F2MinusF1 => df2 - df1,
            CellOutput::Border => dborder,
            CellOutput::CellValue => Vec3::ZERO,
        };
        // In-cell positions move `frequency` cells per domain unit.
        let gradient = in_cell * self.cellular.lattice.frequency;
        if weight == 1.0 {
            (value, gradient)
        } else {
            (self.mean + (value - self.mean) * weight, gradient * weight)
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::CellOutput;
    use crate::hash::hash;
    use crate::lanes::tests::{RUNS, coherent_cells, same_bits};

    /// [`lattice_eval3`] as first written, hashing every corner in full and
    /// skipping the corners outside the kernel by a branch: the reference
    /// its hoisted hashing and gathered corners must match bit for bit.
    fn reference_eval3<const GRADIENT: bool>(
        basis: Basis,
        lattice: Lattice3,
        seed: u64,
        offset: Vec3,
        p: Vec3,
    ) -> (f32, Vec3) {
        let at = lattice.locate(p);
        let mut cell = at.cell;
        let mut f = at.frac + offset;
        for axis in 0..3 {
            if f[axis] >= 1.0 {
                f[axis] -= 1.0;
                cell[axis] += 1;
            }
        }
        let corner = |dx: i64, dy: i64, dz: i64| -> u64 {
            let [x, y, z] = lattice.wrap_cell([cell[0] + dx, cell[1] + dy, cell[2] + dz]);
            hash(seed, &[LATTICE_TAG, key(x), key(y), key(z)])
        };
        match basis {
            Basis::Value => {
                let value = |h: u64| unit_f32(h) * 2.0 - 1.0;
                let c = |dx, dy, dz| value(corner(dx, dy, dz));
                let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
                let (c000, c100, c010, c110) = (c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(1, 1, 0));
                let (c001, c101, c011, c111) = (c(0, 0, 1), c(1, 0, 1), c(0, 1, 1), c(1, 1, 1));
                let x00 = lerp(c000, c100, u);
                let x10 = lerp(c010, c110, u);
                let x01 = lerp(c001, c101, u);
                let x11 = lerp(c011, c111, u);
                let y0 = lerp(x00, x10, v);
                let y1 = lerp(x01, x11, v);
                let result = lerp(y0, y1, w);
                if !GRADIENT {
                    return (result, Vec3::ZERO);
                }
                let (du, dv, dw) = (
                    fade_derivative(f.x),
                    fade_derivative(f.y),
                    fade_derivative(f.z),
                );
                let dx0 = lerp(c100 - c000, c110 - c010, v);
                let dx1 = lerp(c101 - c001, c111 - c011, v);
                let dfx = lerp(dx0, dx1, w) * du;
                let dfy = lerp(x10 - x00, x11 - x01, w) * dv;
                let dfz = (y1 - y0) * dw;
                (result, Vec3::new(dfx, dfy, dfz) * lattice.frequency)
            }
            Basis::Gradient => {
                let mut sum = 0.0;
                let mut gradient = Vec3::ZERO;
                for dz in -1..=2_i64 {
                    for dy in -1..=2_i64 {
                        for dx in -1..=2_i64 {
                            let d = Vec3::new(f.x - dx as f32, f.y - dy as f32, f.z - dz as f32);
                            let r2 = d.length_squared();
                            if r2 >= KERNEL_RADIUS_SQ {
                                continue;
                            }
                            let h = corner(dx, dy, dz);
                            let [gx, gy, gz] = GRADIENTS[(h >> 56) as usize];
                            let value =
                                ((h >> 32) & 0xFF_FFFF) as f32 * (1.0 / 16_777_216.0) * 2.0 - 1.0;
                            let t = 1.0 - r2 / KERNEL_RADIUS_SQ;
                            let t2 = t * t;
                            let term = gx * d.x + gy * d.y + gz * d.z + VALUE_WEIGHT * value;
                            sum += t2 * t2 * term;
                            if GRADIENT {
                                let dk = -8.0 * t2 * t / KERNEL_RADIUS_SQ;
                                gradient += d * (dk * term) + Vec3::new(gx, gy, gz) * (t2 * t2);
                            }
                        }
                    }
                }
                (
                    sum * GRADIENT_SCALE,
                    gradient * GRADIENT_SCALE * lattice.frequency,
                )
            }
        }
    }

    /// [`Fractal3`]'s sum over [`reference_eval3`].
    fn reference_sum<const GRADIENT: bool>(
        fractal: &Fractal3,
        p: Vec3,
        footprint: Footprint,
    ) -> (f32, Vec3) {
        let mean = fractal.mean();
        let mut sum = 0.0;
        let mut gradient = Vec3::ZERO;
        for octave in &fractal.octaves {
            let weight = footprint.band_weight(octave.lattice.max_frequency());
            let value = if weight == 0.0 {
                mean
            } else {
                let (n, dn) = reference_eval3::<GRADIENT>(
                    fractal.basis,
                    octave.lattice,
                    octave.seed,
                    octave.offset,
                    p,
                );
                let (detail, d_detail) = match fractal.kind {
                    FractalKind::Fbm => (n, dn),
                    FractalKind::Ridged => {
                        let r = 1.0 - n.abs();
                        (r * r, dn * (-2.0 * r * n.signum()))
                    }
                };
                if GRADIENT {
                    gradient += d_detail * (weight * octave.amplitude);
                }
                mean + (detail - mean) * weight
            };
            sum += value * octave.amplitude;
        }
        (
            sum / fractal.total_amplitude,
            gradient / fractal.total_amplitude,
        )
    }

    /// [`Cellular3`]'s feature point of the cell `offset` steps from `cell`
    /// as first written, hashing the cell in full: the reference its
    /// hoisted neighborhoods must match bit for bit.
    fn reference_feature(cellular: &Cellular3, cell: [i64; 3], offset: [i8; 3]) -> (Vec3, u64) {
        let [wx, wy, wz] = cellular.lattice.wrap_cell([
            cell[0] + i64::from(offset[0]),
            cell[1] + i64::from(offset[1]),
            cell[2] + i64::from(offset[2]),
        ]);
        let id = hash(cellular.seed, &[FEATURE_TAG, key(wx), key(wy), key(wz)]);
        let jitter = Vec3::new(
            unit_f32(hash(id, &[0])) - 0.5,
            unit_f32(hash(id, &[1])) - 0.5,
            unit_f32(hash(id, &[2])) - 0.5,
        ) * cellular.jitter;
        let corner = Vec3::new(
            f32::from(offset[0]),
            f32::from(offset[1]),
            f32::from(offset[2]),
        );
        (corner + Vec3::splat(0.5) + jitter, id)
    }

    /// Lattices for the bit-identity tests: unwrapped, one cell per period
    /// on every axis or on one, and several cells per period.
    fn test_lattices() -> [(Domain3, Vec3); 5] {
        [
            (Domain3::Space, Vec3::new(0.37, 5.0, 2.0)),
            (Domain3::Space, Vec3::splat(8.0)),
            (Domain3::periodic(1, 1, 1).unwrap(), Vec3::ONE),
            (
                Domain3::periodic(1, 1, 1).unwrap(),
                Vec3::new(8.0, 3.0, 1.0),
            ),
            (
                Domain3::periodic(2, 3, 1).unwrap(),
                Vec3::new(1.5, 2.0, 4.0),
            ),
        ]
    }

    /// Points that reach every branch of the lattice kernels on a lattice of
    /// `frequency`: random points near and far on both signs, points on
    /// cell boundaries and on the kernel's edge (in cells, so exact on the
    /// lattice), signed zeros, tiny and NaN coordinates, and coordinates up
    /// to 2^56 cells, inside the range where the cell arithmetic is defined.
    fn awkward_points(frequency: Vec3) -> Vec<Vec3> {
        let mut points = Vec::new();
        for i in 0..400_u64 {
            let h = hash(0x0073_6f6c_6964, &[i]);
            let unit = |shift: u32| unit_f32(h.rotate_left(shift)) * 2.0 - 1.0;
            let scale = [1.0, 3.0, 1e3, 1e6][(i % 4) as usize];
            points.push(Vec3::new(unit(0), unit(21), unit(42)) * scale);
        }
        // In cells: boundaries, halves (a corner two cells away is then on
        // the kernel's edge, 1.5 cells off), and just below a boundary.
        let below_one = 1.0 - f32::EPSILON;
        let cells = [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, below_one, -below_one, 2.0];
        for &a in &cells {
            for &b in &cells[..5] {
                points.push(Vec3::new(a, b, 0.0) / frequency);
                points.push(Vec3::new(b, 0.0, a) / frequency);
            }
        }
        let far = 2.0_f32.powi(56);
        let edges = [0.0, -0.0, 1e-30, -1e-30, f32::NAN, 1e12, -1e12];
        for v in edges {
            points.extend([
                Vec3::splat(v),
                Vec3::new(v, 0.3, -0.7),
                Vec3::new(-0.7, v, 0.3),
                Vec3::new(0.3, -0.7, v),
            ]);
        }
        for v in [far, -far] {
            points.push(Vec3::new(v, 0.25, -0.25) / frequency.x);
            points.push(Vec3::splat(v) / frequency.max_element());
        }
        points
    }

    #[test]
    fn hoisted_corner_hashes_match_the_reference_bit_for_bit() {
        let offsets = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.25, 0.75),
            Vec3::splat(1.0 - f32::EPSILON),
        ];
        let bits = |(v, g): (f32, Vec3)| [v.to_bits(), g.x.to_bits(), g.y.to_bits(), g.z.to_bits()];
        for (domain, frequency) in test_lattices() {
            let lattice = Lattice3::new(domain, frequency).unwrap();
            let points = awkward_points(frequency);
            for basis in [Basis::Value, Basis::Gradient] {
                for (seed, offset) in [0, 7, u64::MAX].into_iter().zip(offsets) {
                    for &p in &points {
                        assert_eq!(
                            bits(lattice_eval3::<false>(basis, lattice, seed, offset, p)),
                            bits(reference_eval3::<false>(basis, lattice, seed, offset, p)),
                            "{basis:?} {lattice:?} seed {seed} at {p}"
                        );
                        assert_eq!(
                            bits(lattice_eval3::<true>(basis, lattice, seed, offset, p)),
                            bits(reference_eval3::<true>(basis, lattice, seed, offset, p)),
                            "{basis:?} {lattice:?} seed {seed} at {p}, with gradient"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn solid_noise_and_fractals_match_the_reference_bit_for_bit() {
        let footprints =
            [0.0, 1e-6, 1.0 / 256.0, 0.02, 0.07, 0.3, 1e6].map(|w| Footprint::new(w).unwrap());
        let bits = |(v, g): (f32, Vec3)| [v.to_bits(), g.x.to_bits(), g.y.to_bits(), g.z.to_bits()];
        for (domain, frequency) in test_lattices() {
            let points = awkward_points(frequency);
            for basis in [Basis::Value, Basis::Gradient] {
                let noise = Noise3::new(basis, domain, frequency, 3).unwrap();
                let fractals = [FractalKind::Fbm, FractalKind::Ridged].map(|kind| {
                    let params = FractalParams {
                        kind,
                        octaves: 4,
                        ..FractalParams::default()
                    };
                    Fractal3::new(basis, domain, frequency, 11, params).unwrap()
                });
                for footprint in footprints {
                    let weight = footprint.band_weight(noise.lattice.max_frequency());
                    for &p in &points {
                        let reference = if weight == 0.0 {
                            (0.0, Vec3::ZERO)
                        } else {
                            let (n, g) =
                                reference_eval3::<true>(basis, noise.lattice, 3, Vec3::ZERO, p);
                            (n * weight, g * weight)
                        };
                        assert_eq!(
                            bits(noise.eval_gradient(p, footprint)),
                            bits(reference),
                            "{noise:?} at {p} with {footprint:?}"
                        );
                        assert_eq!(
                            noise.eval(p, footprint).to_bits(),
                            reference.0.to_bits(),
                            "{noise:?} at {p} with {footprint:?}"
                        );
                        for fractal in &fractals {
                            assert_eq!(
                                fractal.eval(p, footprint).to_bits(),
                                reference_sum::<false>(fractal, p, footprint).0.to_bits(),
                                "{basis:?} {:?} at {p} with {footprint:?}",
                                fractal.kind
                            );
                            assert_eq!(
                                bits(fractal.eval_gradient(p, footprint)),
                                bits(reference_sum::<true>(fractal, p, footprint)),
                                "{basis:?} {:?} at {p} with {footprint:?}",
                                fractal.kind
                            );
                        }
                    }
                }
            }
        }
    }

    /// `points` as `x`, `y` and `z` columns.
    fn columns(points: &[Vec3]) -> [Vec<f32>; 3] {
        [0, 1, 2].map(|axis| points.iter().map(|p| p[axis]).collect())
    }

    /// The awkward points, then edges a lane meets beside other lanes:
    /// signed zeros, NaN and coordinates of 10¹⁵ on each axis, lattice
    /// boundaries in cells and just either side of them, and negatives.
    fn lane_points(frequency: Vec3) -> Vec<Vec3> {
        let mut points = awkward_points(frequency);
        for v in [0.0, -0.0, f32::NAN, 1e15, -1e15, -3.0, -2.5, -1e-7] {
            points.extend([
                Vec3::new(v, 0.4, 0.6),
                Vec3::new(0.6, v, -0.4),
                Vec3::new(-0.4, 0.6, v),
            ]);
        }
        for cells in [1.0_f32, 2.0, -1.0, -3.0, 17.0] {
            for step in [-1, 0, 1] {
                let p = Vec3::splat(cells) / frequency;
                let nudge = |c: f32| f32::from_bits(c.to_bits().wrapping_add_signed(step));
                points.push(Vec3::from_array(p.to_array().map(nudge)));
            }
        }
        points
    }

    #[test]
    fn lane_kernels_match_the_point_kernel() {
        let offsets = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.25, 0.75),
            Vec3::splat(1.0 - f32::EPSILON),
        ];
        for (domain, frequency) in test_lattices() {
            let lattice = Lattice3::new(domain, frequency).unwrap();
            // First the patterns of cells the kernel reuses a lane's cell
            // across, so that runs begin where they do.
            let mut points: Vec<Vec3> = coherent_cells(lattice.wrap)
                .into_iter()
                .map(|cells| Vec3::from_array(cells) / frequency)
                .collect();
            points.extend(lane_points(frequency));
            let [x, y, z] = columns(&points);
            for basis in [Basis::Value, Basis::Gradient] {
                for (seed, offset) in [0, 7, u64::MAX].into_iter().zip(offsets) {
                    lattice_lanes3(basis, lattice, seed, offset, [&[]; 3], &mut []);
                    for run in RUNS.into_iter().filter(|&run| run > 0) {
                        let mut out = vec![-7.0; points.len()];
                        for (start, out) in (0..).step_by(run).zip(out.chunks_mut(run)) {
                            let span = start..start + out.len();
                            let p = [&x[span.clone()], &y[span.clone()], &z[span]];
                            lattice_lanes3(basis, lattice, seed, offset, p, out);
                        }
                        for (&p, &got) in points.iter().zip(&out) {
                            let want = lattice_eval3::<false>(basis, lattice, seed, offset, p).0;
                            assert!(
                                same_bits(got, want),
                                "{basis:?} {lattice:?} seed {seed} in runs of {run} at {p}: \
                                 {got} for {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn noise_and_fractals_over_lanes_match_their_points() {
        // Footprints that keep, fade (by weights either side of a half) and
        // drop each octave.
        let footprints = [
            0.0,
            1e-6,
            1.0 / 256.0,
            0.02,
            0.05,
            0.07,
            0.085,
            0.2,
            0.3,
            0.4,
            1e6,
        ]
        .map(|w| Footprint::new(w).unwrap());
        for (domain, frequency) in test_lattices() {
            let points = lane_points(frequency);
            let [x, y, z] = columns(&points);
            for basis in [Basis::Value, Basis::Gradient] {
                let noise = Noise3::new(basis, domain, frequency, 3).unwrap();
                let fractals = [FractalKind::Fbm, FractalKind::Ridged].map(|kind| {
                    let params = FractalParams {
                        kind,
                        octaves: 4,
                        ..FractalParams::default()
                    };
                    Fractal3::new(basis, domain, frequency, 11, params).unwrap()
                });
                for footprint in footprints {
                    for run in RUNS.into_iter().filter(|&run| run > 0) {
                        let mut noises = vec![-7.0; points.len()];
                        let mut sums = [(); 2].map(|()| vec![-7.0; points.len()]);
                        for (start, out) in (0..).step_by(run).zip(noises.chunks_mut(run)) {
                            let span = start..start + out.len();
                            let p = [&x[span.clone()], &y[span.clone()], &z[span.clone()]];
                            noise.eval_lanes(p, footprint, out);
                            for (fractal, sums) in fractals.iter().zip(&mut sums) {
                                fractal.eval_lanes(p, footprint, &mut sums[span.clone()]);
                            }
                        }
                        for (i, &p) in points.iter().enumerate() {
                            let want = noise.eval(p, footprint);
                            assert!(
                                same_bits(noises[i], want),
                                "{noise:?} in runs of {run} at {p} with {footprint:?}"
                            );
                            for (fractal, sums) in fractals.iter().zip(&sums) {
                                assert!(
                                    same_bits(sums[i], fractal.eval(p, footprint)),
                                    "{basis:?} {:?} in runs of {run} at {p} with {footprint:?}",
                                    fractal.kind
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn neighborhoods_match_cells_hashed_one_by_one() {
        for (domain, frequency) in test_lattices() {
            for (seed, jitter) in [(9, 0.8), (0, 1.0), (u64::MAX, 0.0)] {
                let cellular = Cellular3::new(domain, frequency, jitter, seed).unwrap();
                for i in 0..96_u64 {
                    let h = hash(0x0063_656c_6c33, &[i]);
                    let scale = [4_i64, 1 << 20, 1 << 40, 1 << 61][(i % 4) as usize];
                    let cell = [
                        h.cast_signed() % scale,
                        (h >> 3).cast_signed() % scale,
                        h.rotate_left(17).cast_signed() % scale,
                    ];
                    let points = cellular.neighborhood(cell);
                    let mut slot = 0;
                    for dz in -3..=3_i8 {
                        for dy in -3..=3_i8 {
                            for dx in -3..=3_i8 {
                                let (point, id) = reference_feature(&cellular, cell, [dx, dy, dz]);
                                assert_eq!(points[slot].1, id, "{domain:?} cell {cell:?}");
                                assert_eq!(
                                    points[slot].0.to_array().map(f32::to_bits),
                                    point.to_array().map(f32::to_bits),
                                    "{domain:?} cell {cell:?}"
                                );
                                slot += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    fn points(count: usize, scale: f32, seed: u64) -> Vec<Vec3> {
        (0..count)
            .map(|i| {
                let h = |axis: u64| unit_f32(hash(seed, &[i as u64, axis])) * scale;
                Vec3::new(h(0), h(1), h(2))
            })
            .collect()
    }

    #[test]
    fn gradients_are_unit_and_off_axis() {
        for [x, y, z] in GRADIENTS {
            assert!(
                (x * x + y * y + z * z - 1.0).abs() < 1e-6,
                "({x}, {y}, {z})"
            );
            assert!(x != 0.0 && y != 0.0 && z != 0.0, "({x}, {y}, {z})");
        }
    }

    #[test]
    fn periodic_solids_repeat_bit_exactly() {
        let domain = Domain3::periodic(2, 1, 3).unwrap();
        let params = FractalParams::default();
        let fields: [&dyn SolidField; 4] = [
            &Noise3::new(Basis::Gradient, domain, Vec3::new(2.0, 3.0, 1.0), 4).unwrap(),
            &Noise3::new(Basis::Value, domain, Vec3::new(1.5, 2.0, 1.0), 4).unwrap(),
            &Fractal3::new(Basis::Gradient, domain, Vec3::splat(1.0), 9, params).unwrap(),
            &Cellular3::new(domain, Vec3::new(2.0, 2.0, 1.0), 1.0, 3)
                .unwrap()
                .output(CellOutput::Border),
        ];
        for field in fields {
            for p in points(64, 2.0, 1) {
                // Dyadic points, so adding the period is exact.
                let p = (p * 64.0).floor() / 64.0;
                let a = field.eval(p, Footprint::POINT);
                let b = field.eval(p + Vec3::new(2.0, -1.0, 6.0), Footprint::POINT);
                assert_eq!(a.to_bits(), b.to_bits(), "repeat at {p}");
            }
        }
    }

    #[test]
    fn noise_stays_in_range_and_fades() {
        for basis in [Basis::Value, Basis::Gradient] {
            let noise = Noise3::new(basis, Domain3::Space, Vec3::splat(3.0), 11).unwrap();
            for p in points(4096, 5.0, 2) {
                let v = noise.eval(p, Footprint::POINT);
                assert!((-1.0..=1.0).contains(&v), "{basis:?} value {v} at {p}");
            }
            let coarse = Footprint::new(0.2).unwrap();
            assert_eq!(noise.eval(Vec3::splat(0.3), coarse), 0.0);
        }
    }

    #[test]
    fn analytic_gradients_match_central_differences() {
        let params = FractalParams::default();
        let fields: [&dyn SolidField; 5] = [
            &Noise3::new(Basis::Gradient, Domain3::Space, Vec3::new(2.0, 3.0, 1.5), 4).unwrap(),
            &Noise3::new(Basis::Value, Domain3::Space, Vec3::new(2.0, 1.0, 3.0), 5).unwrap(),
            &Fractal3::new(Basis::Gradient, Domain3::Space, Vec3::splat(1.0), 9, params).unwrap(),
            &Cellular3::new(Domain3::Space, Vec3::splat(2.0), 1.0, 3)
                .unwrap()
                .output(CellOutput::F1),
            &Cellular3::new(Domain3::Space, Vec3::splat(2.0), 0.8, 7)
                .unwrap()
                .output(CellOutput::F2MinusF1),
        ];
        for field in fields {
            for p in points(48, 3.0, 3) {
                let (value, gradient) = field.eval_gradient(p, Footprint::POINT);
                assert_eq!(value.to_bits(), field.eval(p, Footprint::POINT).to_bits());
                let numeric = central_difference3(field, p, Footprint::new(2e-3).unwrap());
                let error = (gradient - numeric).length() / gradient.length().max(1.0);
                // Cellular fields have creases; skip points straddling one.
                assert!(
                    error < 2e-2 || (gradient - numeric).length() > 1.0,
                    "{p}: {gradient} vs {numeric}"
                );
            }
        }
    }

    /// Each output, with and without its gradient, keeps the bits of the
    /// full lookup, though only a border output searches for the border.
    #[test]
    fn cellular_fields_match_the_full_sample() {
        let domains = [
            (Domain3::Space, Vec3::new(0.37, 5.0, 1.5)),
            (
                Domain3::periodic(2, 3, 1).unwrap(),
                Vec3::new(1.5, 2.0, 3.0),
            ),
        ];
        let footprints = [0.0, 0.12, 0.4].map(|w| Footprint::new(w).unwrap());
        for (domain, frequency) in domains {
            let cellular = Cellular3::new(domain, frequency, 0.8, 5).unwrap();
            for output in [
                CellOutput::F1,
                CellOutput::F2,
                CellOutput::F2MinusF1,
                CellOutput::Border,
                CellOutput::CellValue,
            ] {
                let field = cellular.output(output);
                for p in awkward_points(frequency).into_iter().step_by(3) {
                    let (sample, [df1, df2, dborder]) = cellular.sample::<true>(p, true);
                    assert!(same_bits(
                        select(output, &cellular.sample::<false>(p, true).0),
                        select(output, &sample)
                    ));
                    for footprint in footprints {
                        let weight = footprint.band_weight(cellular.lattice.max_frequency());
                        let fade = |v: f32| {
                            if weight == 0.0 {
                                field.mean
                            } else if weight == 1.0 {
                                v
                            } else {
                                field.mean + (v - field.mean) * weight
                            }
                        };
                        let want = fade(select(output, &sample));
                        let got = field.eval(p, footprint);
                        assert!(
                            same_bits(got, want),
                            "{output:?} at {p} with {footprint:?}: {got} for {want}"
                        );
                        let (value, gradient) = field.eval_gradient(p, footprint);
                        assert!(same_bits(value, want), "{output:?} at {p}, with gradient");
                        if weight > 0.0 {
                            let in_cell = match output {
                                CellOutput::F1 => df1,
                                CellOutput::F2 => df2,
                                CellOutput::F2MinusF1 => df2 - df1,
                                CellOutput::Border => dborder,
                                CellOutput::CellValue => Vec3::ZERO,
                            } * frequency;
                            let want = if weight == 1.0 {
                                in_cell
                            } else {
                                in_cell * weight
                            };
                            for axis in 0..3 {
                                assert!(
                                    same_bits(gradient[axis], want[axis]),
                                    "{output:?} at {p}: gradient {gradient} for {want}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cellular_search_is_exact() {
        // Brute force over a 11³ block agrees with the 7³ search.
        let cellular = Cellular3::new(Domain3::Space, Vec3::ONE, 1.0, 12).unwrap();
        for p in points(200, 4.0, 4) {
            let (sample, _) = cellular.sample::<false>(p, true);
            let at = cellular.lattice.locate(p);
            let mut d: Vec<f32> = Vec::new();
            for dz in -5..=5_i8 {
                for dy in -5..=5_i8 {
                    for dx in -5..=5_i8 {
                        let (point, _) = reference_feature(&cellular, at.cell, [dx, dy, dz]);
                        d.push((point - at.frac).length_squared());
                    }
                }
            }
            d.sort_by(f32::total_cmp);
            assert_eq!(sample.f1, libm::sqrtf(d[0]));
            assert_eq!(sample.f2, libm::sqrtf(d[1]));
        }
    }

    #[test]
    fn ridged_means_match_their_constants() {
        for basis in [Basis::Value, Basis::Gradient] {
            let mut sum = 0.0_f64;
            let samples = points(40_000, 50.0, 5);
            for (i, p) in samples.iter().enumerate() {
                let noise = Noise3::new(basis, Domain3::Space, Vec3::ONE, i as u64 % 8).unwrap();
                let r = 1.0 - noise.eval(*p, Footprint::POINT).abs();
                sum += f64::from(r * r);
            }
            let mean = sum / samples.len() as f64;
            assert!(
                (mean - f64::from(ridged_mean3(basis))).abs() < 0.01,
                "{basis:?}: measured {mean}"
            );
        }
    }
}
