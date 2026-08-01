import { NavLink, Route, Routes } from "react-router-dom";
import { useMe } from "./api/client";
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
            Gantry <span className="tag">BENCH</span>
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
      </Routes>
    </div>
  );
}
