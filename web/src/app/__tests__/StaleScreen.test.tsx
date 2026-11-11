import { act, render, screen } from '@testing-library/react';
import type { LoadedSession } from '@/api/schemas';
import { sessionStore } from '@/core/stores/session';
import { draftStatusStore } from '@/features/draft/draftStatus';
import { setLang } from '@/i18n';
import { StaleScreen } from '../StaleScreen';
import { sessionLoaded } from './fakeServer';

const model = sessionLoaded as unknown as LoadedSession;

beforeEach(() => {
  setLang('tr');
  sessionStore.setState({ model });
});
afterEach(() => {
  sessionStore.setState({ model: null });
  draftStatusStore.setState({ hasDraft: false, tooLarge: false, pendingSha: null });
});

test('down with a draft that autosave keeps current', () => {
  render(<StaleScreen kind="down" hasDraft />);
  expect(screen.getByRole('heading', { name: 'Sunucu kapandı; kesimlerin taslakta' })).toBeInTheDocument();
});

test('down while an earlier draft holds autosave back: new cuts are not claimed to be in the draft', () => {
  draftStatusStore.setState({ hasDraft: true, pendingSha: model.source_sha256 });
  render(<StaleScreen kind="down" hasDraft />);
  expect(screen.queryByText('Sunucu kapandı; kesimlerin taslakta')).toBeNull();
  expect(screen.getByRole('heading', { name: 'Sunucu kapandı; bu oturumdaki yeni kesimler taslağa yazılmadı' })).toBeInTheDocument();
});

test('restarted: the restore hint is conditional on the held state too', () => {
  const view = render(<StaleScreen kind="restarted" hasDraft />);
  expect(screen.getByText('Kesimleriniz yeni oturumda geri yüklenebilir.')).toBeInTheDocument();
  view.unmount();
  act(() => draftStatusStore.setState({ pendingSha: model.source_sha256 }));
  render(<StaleScreen kind="restarted" hasDraft />);
  expect(screen.queryByText('Kesimleriniz yeni oturumda geri yüklenebilir.')).toBeNull();
  expect(screen.getByText(/bu oturumdaki yeni kesimler taslağa yazılmadı/)).toBeInTheDocument();
});
