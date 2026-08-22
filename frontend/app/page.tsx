import type { Metadata } from "next";
import SignalLab from "./SignalLab";

export const metadata: Metadata = {
  title: "Photonic Signal Lab",
  description:
    "Explore optical-fibre distortion and compare causal signal recovery with a linear equalizer, digital ESN, and photonic delay reservoir.",
};

export default function Home() {
  return <SignalLab />;
}
