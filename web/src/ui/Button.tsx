import type { ButtonHTMLAttributes } from 'react';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost';

type Props = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant };

export function Button({ variant = 'secondary', type = 'button', className, ...rest }: Props) {
  return <button type={type} className={`btn btn-${variant}${className ? ` ${className}` : ''}`} {...rest} />;
}
