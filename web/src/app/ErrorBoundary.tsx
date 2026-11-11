import { Component, type ErrorInfo, type ReactNode } from 'react';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { TechDetail } from '@/ui/TechDetail';

function Fallback({ error }: { error: Error }) {
  const t = useT();
  return (
    <main className="stale-screen" role="alert">
      <h1>{t('app.error.title')}</h1>
      <TechDetail>{error.message}</TechDetail>
      <Button variant="primary" onClick={() => window.location.reload()}>
        {t('app.error.reload')}
      </Button>
    </main>
  );
}

/** A render error shows one calm screen instead of a blank page. The message is attacker-influenced: text only. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error): { error: Error } {
    return { error };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(error, info.componentStack);
  }

  override render(): ReactNode {
    return this.state.error === null ? this.props.children : <Fallback error={this.state.error} />;
  }
}
