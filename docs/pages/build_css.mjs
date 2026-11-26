// Compile the doc-builder kit stylesheet for the standalone GitHub Pages site.
//
// doc-builder empties every CSS file in production builds (kit/postcss.config.cjs)
// because huggingface.co injects its own Tailwind. Outside the Hub we need that
// CSS back, so this script runs the kit's own `src/app.css` through Tailwind,
// scanning the already built pages so only the classes in use are emitted.
//
// Usage: node build_css.mjs <doc-builder/kit> <built site dir> <output.css>
// The kit's node_modules must be installed (`npm ci` inside the kit).

import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const [kitArg, siteArg, outArg] = process.argv.slice(2);
if (!kitArg || !siteArg || !outArg) {
	console.error("usage: node build_css.mjs <kit dir> <site dir> <output.css>");
	process.exit(2);
}
const kitDir = path.resolve(kitArg);
const siteDir = path.resolve(siteArg);
const outFile = path.resolve(outArg);

// Resolve postcss and plugins from the kit's node_modules, not from this repo.
const kitRequire = createRequire(path.join(kitDir, "package.json"));
const postcss = kitRequire("postcss");
const tailwind = kitRequire("@tailwindcss/postcss");
const autoprefixer = kitRequire("autoprefixer");

const entry = path.join(kitDir, "src", "app.css");
let css = readFileSync(entry, "utf8");

// The kit scans `./routes`, which only exists while doc-builder builds.
// Scan the built HTML and JS chunks instead.
const sourceLine = '@source "./routes";';
if (!css.includes(sourceLine)) {
	console.error(`expected ${sourceLine} in ${entry}; the kit changed, update this script`);
	process.exit(1);
}
css = css.replace(sourceLine, `@source "${siteDir.replaceAll("\\", "/")}";`);

const result = await postcss([tailwind({ base: kitDir }), autoprefixer]).process(css, {
	from: entry,
	to: outFile,
});
writeFileSync(outFile, result.css);
console.log(`wrote ${outFile} (${(result.css.length / 1024).toFixed(0)} KiB)`);
