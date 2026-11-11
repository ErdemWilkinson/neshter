import { useT } from '@/i18n';

/** One chip per token. The piece is attacker-influenced model vocabulary: it is only ever React text. */
export function TokenStrip({
  tokens,
  index,
  disabled,
  onSelect,
}: {
  tokens: readonly string[];
  /** Position of the chosen token. */
  index: number;
  disabled: boolean;
  onSelect: (index: number) => void;
}) {
  const t = useT();
  return (
    <ul className="token-strip" aria-label={t('map.tokens')}>
      {tokens.map((piece, i) => (
        <li key={i}>
          <button
            type="button"
            className="token-chip"
            aria-pressed={i === index}
            disabled={disabled}
            // A blank piece (a space, a newline) has no visible text for a screen reader to find.
            aria-label={piece.trim() === '' ? t('map.token.blank', { n: i + 1 }) : undefined}
            onClick={() => onSelect(i)}
          >
            {piece}
          </button>
        </li>
      ))}
    </ul>
  );
}
