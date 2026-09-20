"use client";

import { useMemo, useState } from "react";

const layers = [
  { id: "baseline", number: "01", title: "Market expectations", subtitle: "Baseline sentiment before results", status: "Before", value: "72", detail: "Positive" },
  { id: "quality", number: "02", title: "Pulse result quality", subtitle: "What the reported quarter says", status: "Result", value: "—", detail: "Awaiting event" },
  { id: "estimates", number: "03", title: "Actual vs estimates", subtitle: "Consensus surprise, metric by metric", status: "Present", value: "—", detail: "Connect estimates" },
  { id: "verdict", number: "04", title: "AI verdict", subtitle: "Guidance + sector outlook", status: "Forward", value: "—", detail: "Concall pending" },
  { id: "post", number: "05", title: "Post-result sentiment", subtitle: "Market confirmation—or contradiction", status: "Safety", value: "—", detail: "Awaiting result" },
  { id: "setup", number: "06", title: "Actionable setup", subtitle: "TechnoFunda / CANSLIM signals", status: "Setup", value: "—", detail: "Build signal" },
];

const watchlist = [
  { symbol: "RELIANCE", name: "Reliance Industries", price: "1,412.80", change: "+1.24%" },
  { symbol: "HDFCBANK", name: "HDFC Bank", price: "1,008.35", change: "+0.68%" },
  { symbol: "INFY", name: "Infosys", price: "1,542.10", change: "-0.41%" },
  { symbol: "AXISBANK", name: "Axis Bank", price: "1,194.25", change: "+0.92%" },
];

export default function Home() {
  const [selected, setSelected] = useState("baseline");
  const selectedLayer = useMemo(() => layers.find((layer) => layer.id === selected) ?? layers[0], [selected]);

  return (
    <main className="min-h-screen bg-[#070a0f] text-white">
      <div className="mx-auto flex min-h-screen max-w-[1500px] flex-col px-5 py-5 sm:px-8 lg:px-10">
        <header className="flex items-center justify-between border-b border-white/10 pb-5">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-white text-sm font-black text-[#070a0f]">E</div>
            <div><div className="text-[15px] font-semibold tracking-tight">EarningsOS</div><div className="text-[11px] text-white/40">Earnings Intelligence</div></div>
          </div>
          <div className="hidden items-center gap-2 text-xs text-white/50 sm:flex"><span className="h-2 w-2 rounded-full bg-emerald-400" />Research workspace<span className="mx-2 text-white/15">/</span>NSE · NIFTY 50</div>
          <button className="rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-white/70 transition hover:bg-white/[0.08]">Settings</button>
        </header>

        <section className="grid flex-1 gap-8 py-9 lg:grid-cols-[1fr_330px]">
          <div>
            <div className="mb-8 max-w-3xl">
              <div className="mb-3 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.18em] text-blue-300/80"><span className="h-px w-6 bg-blue-400/60" />Connected view</div>
              <h1 className="text-3xl font-semibold tracking-[-0.035em] text-white sm:text-5xl">One earnings event.<br /><span className="text-white/45">Six connected reads.</span></h1>
              <p className="mt-4 max-w-2xl text-sm leading-6 text-white/50 sm:text-[15px]">Move from pre-result expectations to post-result confirmation without hiding the evidence behind a single opaque score.</p>
            </div>

            <div className="mb-7 flex flex-wrap gap-2">
              {["NIFTY 50", "Upcoming", "Recent", "Watchlist"].map((filter, index) => (
                <button key={filter} className={"rounded-full border px-3.5 py-2 text-xs transition " + (index === 0 ? "border-white/20 bg-white text-black" : "border-white/10 bg-white/[0.03] text-white/55 hover:bg-white/[0.07]")}>{filter}</button>
              ))}
            </div>

            <div className="rounded-2xl border border-white/10 bg-[#0c1119] p-5 shadow-2xl shadow-black/20 sm:p-6">
              <div className="flex flex-col justify-between gap-5 border-b border-white/8 pb-5 sm:flex-row sm:items-start">
                <div><div className="mb-2 flex items-center gap-2"><span className="rounded-md bg-white/8 px-2 py-1 font-mono text-[11px] text-white/70">RELIANCE</span><span className="rounded-md bg-blue-400/10 px-2 py-1 text-[10px] font-medium text-blue-300">NEXT EVENT</span></div><h2 className="text-xl font-semibold">Reliance Industries</h2><p className="mt-1 text-xs text-white/40">Earnings event · FY2026 Q2 · reporting period boundary</p></div>
                <div className="text-left sm:text-right"><div className="text-2xl font-semibold tracking-tight">₹1,412.80</div><div className="mt-1 text-xs text-emerald-300">+1.24% today</div></div>
              </div>

              <div className="grid gap-3 py-5 sm:grid-cols-3">
                <Metric label="Baseline" value="72 / 100" note="positive tilt" />
                <Metric label="Articles analysed" value="93" note="rolling news set" />
                <Metric label="Event status" value="Pre-result" note="no leakage" />
              </div>

              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {layers.map((layer) => {
                  const active = layer.id === selected;
                  return (
                    <button key={layer.id} onClick={() => setSelected(layer.id)} className={"group rounded-xl border p-4 text-left transition " + (active ? "border-blue-400/40 bg-blue-400/[0.07]" : "border-white/8 bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04]")}>
                      <div className="mb-5 flex items-start justify-between"><span className="font-mono text-[10px] text-white/30">{layer.number}</span><span className="rounded-full bg-white/6 px-2 py-1 text-[9px] uppercase tracking-wider text-white/40">{layer.status}</span></div>
                      <div className="text-sm font-medium">{layer.title}</div><div className="mt-1 min-h-10 text-xs leading-5 text-white/40">{layer.subtitle}</div>
                      <div className="mt-4 flex items-end justify-between"><span className="text-lg font-semibold">{layer.value}</span><span className="text-[10px] text-white/30">{layer.detail}</span></div>
                    </button>
                  );
                })}
              </div>

              <div className="mt-4 rounded-xl border border-white/8 bg-black/20 p-4">
                <div className="flex items-center justify-between"><div><div className="text-xs font-medium text-white/80">{selectedLayer.title}</div><div className="mt-1 text-[11px] text-white/35">{selectedLayer.subtitle}</div></div><span className="text-[10px] uppercase tracking-wider text-white/30">Evidence panel</span></div>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <Evidence label="Signal" value={selectedLayer.id === "baseline" ? "Positive" : "Not available"} />
                  <Evidence label="Model state" value={selectedLayer.id === "baseline" ? "FinBERT" : "Pending"} />
                  <Evidence label="Data boundary" value="Before result" />
                </div>
              </div>
            </div>
          </div>

          <aside className="lg:pt-[88px]">
            <div className="rounded-2xl border border-white/10 bg-[#0c1119] p-5">
              <div className="flex items-center justify-between"><h3 className="text-sm font-semibold">Watchlist</h3><span className="text-[10px] uppercase tracking-wider text-white/30">Live*</span></div>
              <div className="mt-4 divide-y divide-white/7">
                {watchlist.map((stock) => (
                  <div key={stock.symbol} className="flex items-center justify-between py-4">
                    <div><div className="font-mono text-xs font-medium">{stock.symbol}</div><div className="mt-1 text-[10px] text-white/30">{stock.name}</div></div>
                    <div className="text-right"><div className="text-xs font-medium">₹{stock.price}</div><div className={"mt-1 text-[10px] " + (stock.change.startsWith("+") ? "text-emerald-300" : "text-rose-300")}>{stock.change}</div></div>
                  </div>
                ))}
              </div>
              <div className="mt-2 rounded-lg bg-white/[0.025] p-3 text-[10px] leading-4 text-white/30">*Market values shown here are interface placeholders until the live market-data service is connected.</div>
            </div>

            <div className="mt-4 rounded-2xl border border-white/10 bg-gradient-to-br from-blue-500/[0.10] to-transparent p-5">
              <div className="text-[10px] uppercase tracking-[0.16em] text-blue-300/70">Pipeline</div>
              <div className="mt-3 text-sm font-medium">Pretrained models only</div>
              <p className="mt-2 text-xs leading-5 text-white/40">Screening layer for Chronos, TimesFM, Kronos and FinCast. No custom training required for this product surface.</p>
              <div className="mt-4 flex flex-wrap gap-1.5">{["Chronos-2", "TimesFM 2.5", "Kronos", "FinCast"].map((model) => <span key={model} className="rounded-md border border-white/8 bg-black/15 px-2 py-1 text-[9px] text-white/45">{model}</span>)}</div>
            </div>
          </aside>
        </section>

        <footer className="border-t border-white/8 py-5 text-[10px] text-white/25">EarningsOS · Evidence-first earnings intelligence · UI foundation</footer>
      </div>
    </main>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="rounded-xl border border-white/7 bg-white/[0.025] p-4"><div className="text-[10px] uppercase tracking-wider text-white/30">{label}</div><div className="mt-2 text-sm font-semibold">{value}</div><div className="mt-1 text-[10px] text-white/30">{note}</div></div>;
}

function Evidence({ label, value }: { label: string; value: string }) {
  return <div className="rounded-lg bg-white/[0.025] px-3 py-2.5"><div className="text-[9px] uppercase tracking-wider text-white/25">{label}</div><div className="mt-1 text-xs text-white/65">{value}</div></div>;
}
