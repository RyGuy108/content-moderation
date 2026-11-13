import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "Review queue", description: "modpipe human review console" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
