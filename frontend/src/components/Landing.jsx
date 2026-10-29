import { useEffect } from "react";

const POINTS = ["Cleans your data", "Explains the causes", "Forecasts risk", "Recommends actions"];

// The first screen: what OpsInsight does, and one way in. Scrolling down, a swipe up or Enter also starts.
export default function Landing({ onStart }) {
  useEffect(() => {
    const wheel = (e) => { if (e.deltaY > 20) onStart(); };
    const key = (e) => { if (["Enter", "ArrowDown", "PageDown", " "].includes(e.key)) onStart(); };
    let startY = null;
    const touchStart = (e) => { startY = e.touches[0].clientY; };
    const touchEnd = (e) => { if (startY !== null && startY - e.changedTouches[0].clientY > 40) onStart(); };
    window.addEventListener("wheel", wheel, { passive: true });
    window.addEventListener("keydown", key);
    window.addEventListener("touchstart", touchStart, { passive: true });
    window.addEventListener("touchend", touchEnd);
    return () => {
      window.removeEventListener("wheel", wheel);
      window.removeEventListener("keydown", key);
      window.removeEventListener("touchstart", touchStart);
      window.removeEventListener("touchend", touchEnd);
    };
  }, [onStart]);

  return (
    <section className="landing" aria-labelledby="landing-title">
      <span className="landing-mark" aria-hidden="true" />
      <h1 id="landing-title">OpsInsight</h1>
      <p className="landing-lead">
        Turn your support tickets and sales orders into clear decisions. OpsInsight cleans your data, shows what is
        going wrong, predicts what will fail next, and recommends the actions that fix it first.
      </p>
      <ul className="landing-points">
        {POINTS.map((p) => <li key={p}>{p}</li>)}
      </ul>
      <button className="btn primary landing-start" onClick={onStart}>Get started</button>
      <button className="landing-scroll" onClick={onStart}>
        Scroll to begin
        <svg viewBox="0 0 12 8" width="12" height="8" aria-hidden="true"><path d="M1 1.5 6 6.5l5-5" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
      </button>
    </section>
  );
}
