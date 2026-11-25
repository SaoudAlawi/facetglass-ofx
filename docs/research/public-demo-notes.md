# Public demo notes (clean room)

Source: the public YouTube video "Introducing Mirage: Raytraced prism effects in post"
(uploader: Node Mill, 15:36, https://youtu.be/6e8ZqPVO4iA), watched 2026-11-25 via its
auto-generated captions and 24 sampled frames. Public material only: no binaries,
installers or internal files were inspected. These notes record *what the product is
shown to do*, so FacetGlass can offer comparable capabilities with its own design,
names, presets and code. Do not copy preset names, UI wording or branding from here.

## The key difference from FacetGlass v1

FacetGlass v1 traces a **thin glass sheet**: a flat front face and a tilted back face,
one refraction in and one out. The product in the video traces a **solid 3D glass
object** (a polygon mesh) placed in 3D between a virtual camera and the footage:

- "We're in a 3D space where the shape is between a virtual camera and the original
  footage ... it's actually a full 3D object with the light bending through it if I spin
  it around" (02:58-03:11). Pulling focus onto it shows a solid faceted object (frame at
  01:54: a faceted polyhedron with a 3D rotate gizmo).
- Light bounces inside the object: a "ray bounce limit" (refraction bounces, set to 150
  for the final shot) controls how many refracted faces are visible (09:30-09:44);
  separate "reflection bounces" show reflections inside the shape (09:49-10:00).
- High IOR gives "little crystalline fractures appearing in the sides" (07:45).

That solid-object model (thickness, internal reflection, total internal reflection,
edges catching light) is what reads as "real glass".

## Look features shown

- **Bevel**: specular lines along the edges; controls for width, roughness,
  subdivisions ("rounds off the flat edges") and an angle limit choosing which edges get
  a bevel (06:34-07:06).
- **Reflections**: reflection source can be the source image; roughness spreads the
  highlights; a fast mode (post-process blur, can spill past a face) and a traced mode
  (accurate, slower) (06:21-07:40).
- **Virtual lights**: one or more camera-facing area lights, draggable in the viewer
  with depth in the inspector; colour temperature (Kelvin) and tint; sharp circular
  highlight at low roughness, gradients across faces at higher roughness (08:44-09:28).
- **Chromatic aberration**: two modes, a cheap RGB split that "falls apart" at high
  strength and a continuous (spectral) default (06:03-06:20).
- **Exposure**: master opacity; highlight boost makes bokeh sharper and brighter
  (07:06-07:18).
- **Depth of field**: f-stop and depth (glass distance); focus can be pulled onto the
  object (00:43-00:47, 02:51).
- **Off-screen / world image**: rays that leave the frame see a blurred "world" made
  from the source: spherical wrap (default), mirror, repeat, transparent (rays coming
  back show the source), clamp; a "world texture blur" control; a feather that softens
  the boundary between source and world image (07:52-08:38).
- **World inputs**: a second video input as the world (Fusion only; "second input
  spherical") with its own colour space (12:51-13:44); an imported 360 HDRI/EXR or any
  image as a world texture, with a colour space control (14:23-14:41).

## Shapes

- A large shape library with variations, grouped into menus (basic shapes, user
  imports) (02:11-02:23); e.g. faceted spheres at different subdivision levels.
- **OBJ import** (Wavefront, exported from Blender), with an option to ignore the OBJ's
  shading and set a smooth-shading angle (11:35-12:37).
- Multiple instances can be stacked (two nodes in Fusion) (13:25).

## Workflow and UI

- **Input colour space** must be set first (S-Log, ARRI LogC); it works in scene-linear
  and expects a log-to-display transform after it (00:30-01:07).
- **On-screen gizmo**: drag to move and rotate the object in 3D; a bar for depth, an arc
  for f-stop, an arc for strength (IOR); the gizmo drives the inspector values
  (00:37-00:47, 02:37-02:51).
- **Quality**: draft/high/final; drops to an interactive quality while dragging the
  gizmo; a frame cache for static setups that turns off when anything animates; a global
  "final quality" toggle that switches every instance in Resolve (01:07-01:58, 03:33-03:57).
  Final quality has minimum and maximum steps (adaptive sampling) to remove grain
  (11:11-11:28).
- **Presets and defaults**: save the current setup as a preset, save defaults (colour
  space, technical settings), rescan the preset folder, turn warnings off (03:57-04:29).
- **Recommended order of operations**: glass effect, then lens emulation, then sensor;
  diffusion filters go between glass and lens (04:29-05:01).

## Motion

- Built-in hand-held loops recorded by holding a phone like a real diopter, processed
  into long seamless loops; speed, strength, start time, rotation gain (05:01-05:25).
- Hold at end or ease out to stillness (05:28-05:35).
- **Import a recorded motion**: an iPhone app ("Sensor Recorder") exports an ARKit pose
  CSV; an editor trims, filters, transforms, reorients, smooths per axis, previews the
  loop seam, loops or holds, offsets position; saved as a user animation; keyframed
  animation can be layered on top (10:09-11:08).

## Performance claims

- Preview at 1080p on an M5 MacBook is interactive at a lower quality; high quality has
  a big performance impact during playback (00:56-01:13).
- "It's not the fastest preview ever", so the final shot was rendered to check (14:04).

## Shape library (second look, 2026-11-25)

Sources: high-resolution cue frames of the same public video at 02:12-02:26 and
06:01-06:02 (shape menus open), the NewsShooter article
(https://www.newsshooter.com/2026/11/07/video-village-mirage-raytraced-glass-effects-plugin/:
"a library of 35+ 3D glass shapes, many of which are similar to popular prism
filters from the real world", custom objects can be imported, bevels on any shape),
and public listings of real effect filters (Prism Lens FX product pages and
retailers: kaleidoscope, split diopter, handheld ripple and split ripple, halo and
split halo, ghost, phantom, split subtle kaleidoscope, radiant, starburst, dream).

Kinds of shapes visible in the demo menus (described by type, not by their names):
a honeycomb of hexagonal lenslets on a disc; a bent, stretched slab; concentric
wave rings (small and large); a crystal lens; convex and concave cylinders;
multi-image prisms (whole and half); a pyramid; a dual-focus double image; a
kaleidoscope; water drops on glass; wobbly / wavy glass; a sphere; a "basic
shapes" submenu with the Platonic solids (tetrahedron, cube, octahedron,
dodecahedron, icosahedron) and geodesic spheres at subdivision levels 0-3; a
"user imports" submenu (OBJ).

FacetGlass covers these kinds with its own procedural generators and its own
names (see `fg3d::shapeName`).
