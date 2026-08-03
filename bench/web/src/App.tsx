import { NavLink, Route, Routes } from "react-router-dom";
import { useMe } from "./api/client";
import { Contact } from "./components/Contact";
import { NewSubmission } from "./routes/NewSubmission";
import { SubmissionDetail } from "./routes/SubmissionDetail";
import { Compare } from "./routes/Compare";
import { VerdictPage } from "./routes/VerdictPage";
import { Submissions } from "./routes/Submissions";

export default function App() {
  const me = useMe();
  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            Gantry <span className="tag">BETA</span>
          </div>
          <nav className="topnav">
            <NavLink to="/" end className={({ isActive }) => (isActive ? "on" : "")}>
              Submissions
            </NavLink>
            <NavLink to="/compare" className={({ isActive }) => (isActive ? "on" : "")}>
              Leaderboard
            </NavLink>
          </nav>
          <div className="topbar-right">{me.data ? me.data.org.name : ""}</div>
        </div>
      </header>
      <Routes>
        <Route path="/" element={<Submissions />} />
        <Route path="/submissions/new" element={<NewSubmission />} />
        <Route path="/submissions/:id" element={<SubmissionDetail />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/submissions/:id/verdict" element={<VerdictPage />} />
        {/* Worked examples live on their own path. They render through the same
            screen as a real result, deliberately, but a sample is not one of
            your submissions and its address should not claim otherwise. */}
        <Route path="/samples/:id" element={<SubmissionDetail />} />
        <Route path="/samples/:id/verdict" element={<VerdictPage />} />
      </Routes>
      {/* Outside Routes: reachable from every screen, including the one where
          somebody has just been told their data did not pass. */}
      <Contact />
    </div>
  );
}
