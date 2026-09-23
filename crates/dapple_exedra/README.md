# dapple_exedra

Bakes dapple materials onto exedra surfaces through their texture charts.

A region of an extracted `exedra_mesh::TriMesh` becomes a texel grid over its
UV chart. Each covered texel records the surface point it stands for, placed
in the material's solid space, and a footprint the size of the texel on the
surface. Evaluate those samples with any dapple evaluator (a solid field
through `SolidProgram::eval_chart`, or a planar field at the chart
coordinates), scatter the values back into the grid with dilated padding, and
map the mesh's own UVs onto the image with the `KHR_texture_transform` offset
and scale the bake reports.
