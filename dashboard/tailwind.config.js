/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          base: "#0a1020",
          panel: "#111c2e",
          elevated: "#16233a",
          hover: "#1c2c47",
        },
        edge: {
          subtle: "#1f2f4a",
          default: "#283d5c",
        },
        accent: {
          DEFAULT: "#00e5b8",
          dim: "#00b89a",
        },
        danger: "#ff4757",
        warning: "#ffa726",
        info: "#3b82f6",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      keyframes: {
        "live-pulse": {
          "0%,100%": { opacity: "1" },
          "50%": { opacity: "0.4" },
        },
        "state-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "live-pulse": "live-pulse 1.6s ease-in-out infinite",
        "state-in": "state-in 0.3s ease-out",
      },
    },
  },
  plugins: [],
};
