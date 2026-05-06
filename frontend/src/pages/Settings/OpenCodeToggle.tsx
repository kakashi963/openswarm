import React from "react";

// Example settings toggle for OpenCode vs legacy
interface Props {
  provider: "anthropic" | "opencode";
  onChange: (p: "anthropic" | "opencode") => void;
}

export const OpenCodeToggle: React.FC<Props> = ({ provider, onChange }) => {
  return (
    <div className="flex items-center gap-3 p-4 bg-zinc-900 rounded-xl">
      <div>
        <div className="font-semibold">AI Provider</div>
        <div className="text-sm text-zinc-400">Choose backend</div>
      </div>
      <div className="flex gap-2 ml-auto">
        <button
          onClick={() => onChange("anthropic")}
          className={`px-4 py-1.5 rounded-lg text-sm ${provider === "anthropic" ? "bg-white text-black" : "bg-zinc-800"}`}
        >
          Anthropic (legacy)
        </button>
        <button
          onClick={() => onChange("opencode")}
          className={`px-4 py-1.5 rounded-lg text-sm ${provider === "opencode" ? "bg-emerald-600 text-white" : "bg-zinc-800"}`}
        >
          OpenCode (free models)
        </button>
      </div>
    </div>
  );
};
