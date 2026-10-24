// Ambient typings for the react-dom/client entry point, which the component
// specs in tests/ import. @types/react-dom is not installed, and this shim
// covers only the surface those specs use. No top-level import or export here,
// so this file stays a global declaration.
declare module 'react-dom/client' {
  import type { ReactNode } from 'react'

  export interface Root {
    render(children: ReactNode): void
    unmount(): void
  }

  export function createRoot(container: Element | DocumentFragment): Root
}
