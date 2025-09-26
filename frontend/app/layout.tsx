import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import './globals.css'
import "@copilotkit/react-ui/styles.css";
import { CopilotKit } from "@copilotkit/react-core";
export const metadata: Metadata = {
  title: 'Shopping Assistant',
  description: 'Powered by LangGraph and CopilotKit',
  generator: 'v0.app',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <style>{`
html {
  font-family: ${GeistSans.style.fontFamily};
  --font-sans: ${GeistSans.variable};
  --font-mono: ${GeistMono.variable};
}
        `}</style>
      </head>
      <body>
        <CopilotKit runtimeUrl='/api/copilotkit' showDevConsole={false} agent='shopping_agent_crewai'>
          {children}
        </CopilotKit>
      </body>
    </html>
  )
}
