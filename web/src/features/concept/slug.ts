const TURKISH: Record<string, string> = {
  ı: 'i',
  ş: 's',
  ğ: 'g',
  ü: 'u',
  ö: 'o',
  ç: 'c',
  İ: 'I',
  Ş: 'S',
  Ğ: 'G',
  Ü: 'U',
  Ö: 'O',
  Ç: 'C',
};

/** A concept id the server accepts (`^[A-Za-z0-9._-]{1,64}$`): Turkish letters transliterated, spaces to `-`, the rest dropped. */
export function slugify(name: string): string {
  const slug = name
    .replace(/[ıŞşĞğÜüÖöÇçİ]/g, (letter) => TURKISH[letter] ?? letter)
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^A-Za-z0-9._-]/g, '')
    .slice(0, 64);
  return slug === '' ? 'kavram' : slug;
}
