import { useId, useState, type FormEvent } from 'react';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';

const ABSOLUTE = /^(?:[A-Za-z]:[\\/]|\\\\|\/)/;

/** An absolute folder path typed by hand. No `name` is sent: the server names the session after the folder. */
export function PathField({ onOpen }: { onOpen: (path: string) => void }) {
  const t = useT();
  const inputId = useId();
  const errorId = useId();
  const [value, setValue] = useState('');
  const [invalid, setInvalid] = useState(false);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const path = value.trim();
    if (!ABSOLUTE.test(path)) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    onOpen(path);
  };

  return (
    <form className="path-field" onSubmit={submit}>
      <label htmlFor={inputId}>{t('models.path.label')}</label>
      <div className="path-field-row">
        <input
          id={inputId}
          type="text"
          value={value}
          spellCheck={false}
          autoComplete="off"
          aria-invalid={invalid || undefined}
          aria-describedby={invalid ? errorId : undefined}
          onChange={(event) => setValue(event.target.value)}
        />
        <Button type="submit" variant="primary">
          {t('models.open')}
        </Button>
      </div>
      {invalid && (
        <p id={errorId} role="alert">
          {t('models.path.invalid')}
        </p>
      )}
    </form>
  );
}
