import js from '@eslint/js';
import { defineConfig, globalIgnores } from 'eslint/config';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

// Scoped overrides use '**/'-prefixed globs on purpose: @eslint/config-array's Windows path.relative lowercases
// whole paths and, for this repo folder (it contains 'İ'), returns a mangled base-relative path such as
// 'c<sep>core<sep>zod.ts'. A literal 'src/core/**' glob then never matches; '**/core/**' matches either way.

const LOCAL_STORAGE = {
  name: 'localStorage',
  message: 'Nothing is persisted client-side: drafts and prefs live on the server (.state/).',
};
const SESSION_STORAGE = {
  name: 'sessionStorage',
  message: 'Use getSessionStorage() from src/app/sessionStorage.ts.',
};

const restrictedProperties = (names) =>
  ['window', 'globalThis', 'self'].flatMap((object) =>
    names.map((property) => ({ object, property, message: `${object}.${property} is not allowed here.` })),
  );

const restrictedSyntax = [
  {
    selector: "JSXAttribute[name.name='dangerouslySetInnerHTML']",
    message: 'Model output and names are attacker-influenced: render them as React text.',
  },
  {
    selector: "AssignmentExpression[left.property.name='innerHTML']",
    message: 'Model output and names are attacker-influenced: render them as React text.',
  },
];

function restrictedImports({ zod, core }) {
  const paths = [
    { name: '@react-three/drei', message: 'No three.js / R3F / drei in Plan 2.' },
    { name: 'i18next', message: 'Own t() in src/i18n.' },
    { name: 'react-i18next', message: 'Own t() in src/i18n.' },
  ];
  if (zod) paths.push({ name: 'zod', message: "Import { z } from '@/core/zod' (jitless config runs there)." });
  if (core) {
    for (const name of ['react', 'react-dom', 'zustand/react']) {
      paths.push({ name, message: 'src/core is framework-free (vanilla stores only).' });
    }
  }
  return ['error', { paths, patterns: [{ group: ['*?worker&inline'], message: 'Blob workers violate the CSP.' }] }];
}

const nodeGlobals = Object.fromEntries(
  ['process', 'console', 'Buffer', 'URL', 'setTimeout', 'clearTimeout'].map((n) => [n, 'readonly']),
);

export default defineConfig([
  globalIgnores(['node_modules/**', 'playwright-report/**', 'test-results/**', 'scripts/__tests__/__fixtures__/**']),
  js.configs.recommended,
  tseslint.configs.strict,
  {
    files: ['**/*.{ts,tsx}'],
    extends: [reactHooks.configs.flat.recommended, jsxA11y.flatConfigs.recommended],
  },
  {
    files: ['**/*.{js,mjs}'],
    languageOptions: { globals: nodeGlobals },
  },
  {
    rules: {
      'no-restricted-globals': ['error', LOCAL_STORAGE, SESSION_STORAGE],
      'no-restricted-properties': ['error', ...restrictedProperties(['localStorage', 'sessionStorage'])],
      'no-restricted-syntax': ['error', ...restrictedSyntax],
      'no-restricted-imports': restrictedImports({ zod: true, core: false }),
    },
  },
  {
    files: ['**/core/**/*.{ts,tsx}'],
    rules: { 'no-restricted-imports': restrictedImports({ zod: true, core: true }) },
  },
  {
    files: ['**/core/zod.ts'],
    rules: { 'no-restricted-imports': restrictedImports({ zod: false, core: true }) },
  },
  {
    files: ['**/app/sessionStorage.ts'],
    rules: {
      'no-restricted-globals': ['error', LOCAL_STORAGE],
      'no-restricted-properties': ['error', ...restrictedProperties(['localStorage'])],
    },
  },
  {
    files: ['**/__tests__/**/*.{ts,tsx}', '**/e2e/**/*.ts'],
    rules: {
      'no-restricted-globals': ['error', SESSION_STORAGE],
      'no-restricted-properties': ['error', ...restrictedProperties(['sessionStorage'])],
    },
  },
]);
