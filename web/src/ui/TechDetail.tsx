import type { ReactNode } from 'react';
import { useT } from '@/i18n';

/** The one place server `detail` text may appear (as React text, collapsed by default). */
export function TechDetail({ children }: { children: ReactNode }) {
  const t = useT();
  return (
    <details className="tech-detail">
      <summary>{t('ui.techDetail')}</summary>
      <pre>{children}</pre>
    </details>
  );
}
