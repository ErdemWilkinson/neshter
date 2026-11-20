import type { Lang } from '@/core/stores/ui';

/**
 * Job progress messages are written in Turkish by the server (and the phone's job runner). The Turkish UI shows them
 * as they are; the English UI translates the known ones here. An unknown message is shown unchanged, never dropped.
 */
const FIXED: Record<string, string> = {
  'SHA-256 hesaplanıyor': 'Computing SHA-256',
  'SHA-256 hazır': 'SHA-256 ready',
  'Ağırlıklar okunuyor': 'Reading the weights',
  'Ağırlıklar okundu': 'Weights read',
  'Dtype ve cihaz ayarlandı': 'Dtype and device set',
  'Seçili müdahalenin etkisi ölçüldü': 'Effect of the selected cut measured',
  'Perplexity ölçülüyor': 'Measuring perplexity',
  'Metrikler hazır': 'Metrics ready',
  'İndirme başlıyor': 'Starting the download',
  'Önceki model bırakılıyor': 'Releasing the previous model',
  'Model yükleniyor': 'Loading the model',
  'Model açılıyor': 'Opening the model',
  'Gerçek etki hesaplanıyor': 'Measuring the real effect',
  'Orijinal cevaplar hesaplanıyor': 'Computing the original answers',
  'Kavram aranıyor': 'Searching for the behavior',
  'KL ve perplexity ölçülüyor': 'Measuring KL and perplexity',
  'Kaynak model doğrulanıyor': 'Checking the source model',
  'Head gücü farkı hesaplandı': 'Head strength difference computed',
  'Ölçüm tamamlandı': 'Measurement done',
  'Ağırlıklara gömülüyor': 'Baking into the weights',
  'Ağırlıklar yazılıyor': 'Writing the weights',
  'Kayıt tamamlanıyor': 'Finishing the save',
  Kaydedildi: 'Saved',
  "GGUF'a dönüştürülüyor ve nicelendiriliyor": 'Converting to GGUF and quantizing',
  "Kart, tarif ve lisans GGUF'un yanına taşınıyor": 'Moving the card, recipe and license next to the GGUF',
  'GGUF hazır': 'GGUF ready',
  "Ollama'ya ekleniyor": 'Adding to Ollama',
  'İleri geçiş': 'Forward pass',
  'Geri geçiş': 'Backward pass',
  'Bağ ağırlıkları': 'Edge weights',
  Nehir: 'River',
  'Grafik hazır': 'Graph ready',
};

const PATTERNS: [RegExp, string][] = [
  [/^Rastgele kontrol (\d+)\/(\d+) \(erken durabilir\)$/, 'Random control $1/$2 (may stop early)'],
  [/^Rastgele kontrol (\d+)\/(\d+)$/, 'Random control $1/$2'],
  [/^(\d+) \/ (\d+) MB indirildi$/, '$1 / $2 MB downloaded'],
  [/^Süpürme: (\d+) yön silindi$/, 'Sweep: $1 directions removed'],
  [/^(\d+)\/(\d+) istek işlendi$/, '$1/$2 prompts processed'],
  [/^(\d+)\/(\d+) atıf geçişi$/, '$1/$2 attribution passes'],
  [/^(\d+)\/(\d+) head ölçüldü$/, '$1/$2 heads measured'],
  [/^(\d+)\/(\d+) adım$/, '$1/$2 steps'],
  [/^İleri geçiş: katman (\d+)\/(\d+)$/, 'Forward pass: layer $1/$2'],
  [/^Geri geçiş: katman (\d+)\/(\d+)$/, 'Backward pass: layer $1/$2'],
  [/^Bağ ağırlıkları: (\d+)\/(\d+) düğüm$/, 'Edge weights: $1/$2 nodes'],
];

export function jobMessage(message: string, lang: Lang): string {
  if (lang === 'tr' || message === '') return message;
  if (Object.hasOwn(FIXED, message)) return FIXED[message] as string;
  for (const [pattern, english] of PATTERNS) if (pattern.test(message)) return message.replace(pattern, english);
  return message;
}
