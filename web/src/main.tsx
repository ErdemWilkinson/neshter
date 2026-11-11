import './core/zod';
import './styles/tokens.css';
import './styles/fonts.css';
import './styles/base.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';

const container = document.getElementById('root');
if (!container) throw new Error('#root missing');
createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
