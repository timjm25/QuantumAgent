import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "QuantumAgent — AI Platform for Quantum Research",
  description:
    "Multi-agent AI system for autonomous quantum computing and quantum chemistry research",
  keywords: [
    "quantum computing",
    "AI research",
    "multi-agent",
    "VQE",
    "quantum chemistry",
    "autonomous research",
  ],
  authors: [{ name: "QuantumAgent Team" }],
  openGraph: {
    title: "QuantumAgent",
    description: "AI Platform for Autonomous Quantum Research",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-gray-950 text-gray-100 font-sans antialiased">
        <div className="flex h-screen overflow-hidden">
          {/* Sidebar */}
          <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col">
            <div className="p-6 border-b border-gray-800">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-violet-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
                  QA
                </div>
                <div>
                  <h1 className="font-bold text-white">QuantumAgent</h1>
                  <p className="text-xs text-gray-400">AI Research Platform</p>
                </div>
              </div>
            </div>
            <nav className="flex-1 p-4 space-y-1">
              {[
                { href: "/", label: "Dashboard", icon: "⚡" },
                { href: "/experiments", label: "Experiments", icon: "🔬" },
                { href: "/quantum", label: "Circuits", icon: "⭕" },
                { href: "/chemistry", label: "Chemistry", icon: "⚗️" },
                { href: "/literature", label: "Literature", icon: "📚" },
                { href: "/knowledge-graph", label: "Knowledge Graph", icon: "🕸️" },
                { href: "/publications", label: "Publications", icon: "📄" },
              ].map((item) => (
                <a
                  key={item.href}
                  href={item.href}
                  className="flex items-center space-x-3 px-3 py-2 rounded-lg text-gray-300 hover:bg-gray-800 hover:text-white transition-colors"
                >
                  <span>{item.icon}</span>
                  <span className="text-sm">{item.label}</span>
                </a>
              ))}
            </nav>
            <div className="p-4 border-t border-gray-800">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-xs text-gray-400">12 agents active</span>
              </div>
            </div>
          </aside>

          {/* Main content */}
          <main className="flex-1 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
