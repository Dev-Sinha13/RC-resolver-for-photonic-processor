import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const incoming = await headers();
  const host = incoming.get("x-forwarded-host") ?? incoming.get("host") ?? "localhost:3000";
  const protocol = incoming.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const base = new URL(`${protocol}://${host}`);
  const title = "Photonic Signal Lab";
  const description = "Watch a damaged light signal become readable again in an interactive optical receiver laboratory.";
  const image = new URL("/og.png", base).toString();
  return {
    metadataBase: base,
    title: { default: title, template: "%s · Photonic Signal Lab" },
    description,
    icons: { icon: "/favicon.svg" },
    openGraph: { title, description, images: [{ url: image, width: 1731, height: 909 }] },
    twitter: { card: "summary_large_image", title, description, images: [image] },
  };
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
