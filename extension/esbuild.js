const esbuild = require("esbuild");

const isProduction = process.argv.includes("--production");
const isWatch = process.argv.includes("--watch");

/** @type {import('esbuild').BuildOptions} */
const buildOptions = {
    entryPoints: ["src/extension.ts"],
    bundle: true,
    outfile: "dist/extension.js",
    external: ["vscode"],
    format: "cjs",
    platform: "node",
    target: "node18",
    sourcemap: !isProduction,
    minify: isProduction,
    logLevel: "info",
};

if (isWatch) {
    esbuild.context(buildOptions).then((ctx) => {
        ctx.watch();
        console.log("Watching for changes...");
    });
} else {
    esbuild.build(buildOptions).catch(() => process.exit(1));
}
