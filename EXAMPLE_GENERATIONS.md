# Example Generations

These examples are prompt and setting recipes for repeatable testing.

## Top down knight

Tab: Generate

Prompt: `small knight with blue cape, top-down RPG sprite, clean outline, limited palette`

Settings:

- Asset type: character
- Size: 64x64 standard character
- Colors: 24
- Transparent background: on
- Seed: 1001

## State variant object

Tab: Objects

Input: choose a saved object or upload a PNG.

Edit: `make it mossy, cracked, and ancient`

Settings:

- Seed: 2104
- LoRA: none or compatible pixel art LoRA

## UI pack recipe

Tab: Recipes

Recipe: UI pack

Theme: `crystal forest RPG interface`

Settings:

- Count: 6
- Size: 64
- Seed: 7007

Expected outputs:

- Buttons
- Slots
- Health bars
- Panel frames
- Metadata manifest

## Four direction object

Tab: Objects

Description: `wooden treasure chest with brass trim and red cloth lining`

Settings:

- View: low top-down
- Size: 64
- Seed: 3301

Expected outputs:

- One sprite sheet
- Direction PNG files
- Metadata JSON

## Character walk test

Tab: Animation

Prompt profile: a clear front facing canonical sprite works best.

Settings:

- Animation: walk
- Directions: 4 or 8
- Frames per direction: default
- Seed: keep fixed across reruns

Expected behavior:

- Mirrored paired directions should preserve identity better.
- Direction names are saved in metadata for importer tooling.
