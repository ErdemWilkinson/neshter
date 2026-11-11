import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { JobState } from '@/api/schemas';
import { setLang } from '@/i18n';
import { Dialog } from '../Dialog';
import { ErrorCard } from '../ErrorCard';
import { LiveRegion, announce } from '../LiveRegion';
import { ProgressCard } from '../ProgressCard';
import { Segmented } from '../Segmented';

beforeEach(() => setLang('tr'));
afterEach(() => vi.useRealTimers());

test('ErrorCard shows the message chosen by code', () => {
  render(<ErrorCard code="busy" />);
  expect(screen.getByRole('alert')).toHaveTextContent('Model meşgul');
  expect(document.querySelector('details')).toBeNull();
});

test('ErrorCard never shows detail outside the disclosure', () => {
  render(<ErrorCard code="internal" detail="Traceback: secret detail" />);
  const detail = screen.getByText('Traceback: secret detail');
  const disclosure = detail.closest('details');
  expect(disclosure).not.toBeNull();
  expect(disclosure).not.toHaveAttribute('open');
  expect(screen.getByText('Teknik ayrıntı').closest('summary')).not.toBeNull();
  // everything outside the disclosure is the code's message only
  expect(screen.getByRole('alert').querySelector('p')).toHaveTextContent('Beklenmeyen bir hata oldu.');
});

test('ErrorCard renders markup in detail as text and stringifies non-string detail', () => {
  render(<ErrorCard code="invalid_input" detail={[{ msg: '<img src=x onerror=alert(1)>' }]} />);
  expect(document.querySelector('img')).toBeNull();
  expect(screen.getByText(/<img src=x onerror=alert\(1\)>/)).toBeInTheDocument();
});

test('Dialog: Esc (the cancel event) calls onCancel and does not close it by itself', () => {
  const onCancel = vi.fn();
  render(
    <Dialog open title="Başlık" onCancel={onCancel} actions={<button type="button">Tamam</button>}>
      <p>gövde</p>
    </Dialog>,
  );
  const dialog = document.querySelector('dialog');
  expect(dialog).not.toBeNull();
  const event = new Event('cancel', { cancelable: true });
  act(() => {
    dialog?.dispatchEvent(event);
  });
  expect(onCancel).toHaveBeenCalledTimes(1);
  expect(event.defaultPrevented).toBe(true);
  expect(dialog).toHaveAttribute('open');
});

test('Dialog: opens with focus on the first action and is named by its title', () => {
  render(
    <Dialog
      open
      title="Başlık"
      onCancel={() => undefined}
      actions={
        <>
          <button type="button">Kaydet</button>
          <button type="button">Vazgeç</button>
        </>
      }
    >
      <button type="button">gövdedeki</button>
    </Dialog>,
  );
  expect(screen.getByRole('button', { name: 'Kaydet' })).toHaveFocus();
  expect(document.querySelector('dialog')).toHaveAttribute('aria-labelledby', screen.getByRole('heading', { name: 'Başlık' }).id);
});

test('Dialog: open=false keeps it closed, and closing follows the prop', () => {
  const { rerender } = render(<Dialog open={false} title="T" onCancel={() => undefined} actions={<button type="button">x</button>} />);
  expect(document.querySelector('dialog')).not.toHaveAttribute('open');
  rerender(<Dialog open title="T" onCancel={() => undefined} actions={<button type="button">x</button>} />);
  expect(document.querySelector('dialog')).toHaveAttribute('open');
  rerender(<Dialog open={false} title="T" onCancel={() => undefined} actions={<button type="button">x</button>} />);
  expect(document.querySelector('dialog')).not.toHaveAttribute('open');
});

const job = (patch: Partial<JobState>): JobState => ({
  id: 'j',
  kind: 'hunt',
  state: 'running',
  progress: 0,
  message: 'Ölçülüyor',
  result: null,
  error: null,
  error_code: null,
  started_at: '2026-11-11T10:00:00.000Z',
  ...patch,
});

test('ProgressCard hides ETA below 5% progress and shows it from 5%', () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-11-11T10:01:00.000Z'));
  const { rerender } = render(<ProgressCard job={job({ progress: 0.04 })} onCancel={() => undefined} />);
  expect(screen.getByText('Geçen süre: 1 dk 0 sn')).toBeInTheDocument();
  expect(screen.queryByText(/Tahmini kalan süre/)).toBeNull();
  rerender(<ProgressCard job={job({ progress: 0.5 })} onCancel={() => undefined} />);
  expect(screen.getByText('Tahmini kalan süre: 1 dk 0 sn')).toBeInTheDocument();
  expect(screen.getByRole('progressbar', { name: 'İlerleme' })).toHaveValue(0.5);
  expect(screen.getByText('%50')).toBeInTheDocument();
});

test('ProgressCard shows no time at all for a job that has not started, and İptal cancels', async () => {
  const onCancel = vi.fn();
  render(<ProgressCard job={job({ state: 'queued', started_at: null, progress: 0.9, message: '<b>x</b>' })} onCancel={onCancel} />);
  expect(screen.queryByText(/Geçen süre/)).toBeNull();
  expect(screen.queryByText(/Tahmini/)).toBeNull();
  expect(screen.getByText('<b>x</b>')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'İptal' }));
  expect(onCancel).toHaveBeenCalledTimes(1);
});

test('ProgressCard ticks the elapsed time', () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-11-11T10:00:05.000Z'));
  render(<ProgressCard job={job({})} onCancel={() => undefined} />);
  expect(screen.getByText('Geçen süre: 5 sn')).toBeInTheDocument();
  act(() => {
    vi.advanceTimersByTime(3000);
  });
  expect(screen.getByText('Geçen süre: 8 sn')).toBeInTheDocument();
});

test('LiveRegion announces politely, and the same text again', () => {
  render(<LiveRegion />);
  const region = screen.getByRole('status');
  expect(region).toHaveAttribute('aria-live', 'polite');
  act(() => announce('Ağ haritası'));
  expect(region).toHaveTextContent('Ağ haritası');
  const first = region.firstElementChild;
  act(() => announce('Ağ haritası'));
  expect(region.firstElementChild).not.toBe(first); // a new node, so assistive tech reads it again
});

test('Segmented is a labelled radio group that reports the choice', async () => {
  const onChange = vi.fn();
  render(
    <Segmented
      label="Görünüm"
      value="2d"
      onChange={onChange}
      options={[
        { value: '2d', label: '2D' },
        { value: 'list', label: 'Liste' },
      ]}
    />,
  );
  expect(screen.getByRole('group', { name: 'Görünüm' })).toBeInTheDocument();
  expect(screen.getByRole('radio', { name: '2D' })).toBeChecked();
  fireEvent.click(screen.getByRole('radio', { name: 'Liste' }));
  expect(onChange).toHaveBeenCalledWith('list');
});
