import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import GameSimulator from "./components/GameSimulator";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <GameSimulator />
  </StrictMode>,
);
