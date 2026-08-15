import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { defineConfig, type Plugin, type ViteDevServer } from "vite";
import react from "@vitejs/plugin-react";

// Dev-only middleware: the client posts structured match events here and we
// append them to match-logs/<gameId>.jsonl so games can be analysed offline.
function matchLogger(): Plugin {
  return {
    name: "match-logger",
    apply: "serve",
    configureServer(server: ViteDevServer) {
      const dir = join(__dirname, "match-logs");
      mkdirSync(dir, { recursive: true });
      server.middlewares.use("/__matchlog", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end();
          return;
        }
        let body = "";
        req.on("data", (chunk) => { body += chunk; });
        req.on("end", () => {
          try {
            const { gameId, lines } = JSON.parse(body) as { gameId: string; lines: unknown[] };
            const safeId = String(gameId).replace(/[^a-z0-9-]/gi, "_");
            appendFileSync(join(dir, `match-${safeId}.jsonl`), lines.map((line) => JSON.stringify(line)).join("\n") + "\n");
            res.end("ok");
          } catch {
            res.statusCode = 400;
            res.end();
          }
        });
      });
    },
  };
}

export default defineConfig({
  base: "/shadowversePt/",
  plugins: [react(), matchLogger()],
  server: {
    watch: { ignored: ["**/match-logs/**"] },
  },
});
