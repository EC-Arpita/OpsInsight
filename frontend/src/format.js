// Turning raw JSON values into readable text, in one place.

export const DOMAINS = {
    incidents: {
      name: "Support tickets", item: "ticket", items: "tickets", countKey: "tickets",
      rateKey: "breach_rate", problemCount: "breaches", problem: "SLA breach", problemShare: "share_of_all_breaches",
    },
    sales: {
      name: "Sales orders", item: "order", items: "orders", countKey: "orders",
      rateKey: "late_rate", problemCount: "late_orders", problem: "late delivery", problemShare: "share_of_all_late",
    },
  };
  
  const nf = new Intl.NumberFormat("en-IN");
  
  export function fmtValue(value, unit) {
    if (value === null || value === undefined) return "–";
    switch (unit) {
      case "ratio": return `${(value * 100).toFixed(1)}%`;
      case "hours": return `${value.toFixed(1)} h`;
      case "days": return `${value.toFixed(1)} days`;
      case "BRL": return value >= 1e6 ? `R$ ${(value / 1e6).toFixed(2)}M` : `R$ ${nf.format(Math.round(value))}`;
      case "stars (1-5)": return `${value.toFixed(2)} ★`;
      default: return nf.format(value);
    }
  }
  
  export const pct = (v, d = 0) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`);
  export const num = (v) => (v === null || v === undefined ? "–" : nf.format(v));
  
  export function fmtDate(iso) {
    if (!iso) return "–";
    return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
  }
  
  export const STATUS_TEXT = { good: "On track", warning: "Watch", critical: "Act now", info: "" };