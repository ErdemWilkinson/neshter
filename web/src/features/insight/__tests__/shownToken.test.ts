import { shownToken } from '../LensPanel';

test('a token is shown on one line', () => {
  expect(shownToken(' ')).toBe('␣');
  expect(shownToken('?"\n\n%')).toBe('?"↵↵%');
  expect(shownToken('a\r\nb\tc')).toBe('a↵b⇥c');
  expect(shownToken(' en')).toBe(' en');
});
