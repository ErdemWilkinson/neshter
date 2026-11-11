import { z } from 'zod';

// zod's default JIT probe uses `new Function`, which the server CSP (no 'unsafe-eval') reports as a violation.
z.config({ jitless: true });

export { z };
