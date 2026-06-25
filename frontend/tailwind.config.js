/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        gray: { 950: "#030712" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  safelist: [
    { pattern: /bg-(violet|blue|green|orange|red)-(400|500|600|700|900)/ },
    { pattern: /text-(violet|blue|green|orange|red)-(300|400|500)/ },
    { pattern: /border-(violet|blue|green|orange|red)-(700)/ },
  ],
  plugins: [],
};
