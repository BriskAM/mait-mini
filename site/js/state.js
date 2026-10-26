// Tiny shared state: the abstention threshold is used by both the live demo and the results charts.
const listeners = new Set();
const state = { threshold: 0.87 };      // default is close to the notebook's value for 99% accuracy (0.866)

export const getThreshold = () => state.threshold;
export function setThreshold(v) { state.threshold = Math.min(0.999, Math.max(0.3, v)); for (const fn of listeners) fn(state.threshold); }
export function onThreshold(fn) { listeners.add(fn); }
