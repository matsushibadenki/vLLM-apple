const header = document.querySelector('[data-header]');
const menuButton = document.querySelector('[data-menu-button]');
const nav = document.querySelector('[data-nav]');

const closeMenu = () => {
  nav?.classList.remove('open');
  menuButton?.setAttribute('aria-expanded', 'false');
};

menuButton?.addEventListener('click', () => {
  const open = nav?.classList.toggle('open');
  menuButton.setAttribute('aria-expanded', String(Boolean(open)));
});
nav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));

const updateHeader = () => header?.classList.toggle('scrolled', window.scrollY > 12);
updateHeader();
window.addEventListener('scroll', updateHeader, { passive: true });

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('visible'));
}

const modelSelect = document.querySelector('[data-model-select]');
const modelLabel = document.querySelector('[data-model-label]');
modelSelect?.addEventListener('change', () => {
  if (modelLabel) modelLabel.textContent = modelSelect.value;
});

const optimizeButton = document.querySelector('[data-optimize]');
const optimizeLabel = document.querySelector('[data-optimize-label]');
const memoryStatus = document.querySelector('[data-memory-status]');
optimizeButton?.addEventListener('click', () => {
  optimizeButton.disabled = true;
  if (optimizeLabel) optimizeLabel.textContent = '正在检查 Mac…';
  if (memoryStatus) memoryStatus.textContent = '正在测量内存';
  window.setTimeout(() => {
    optimizeButton.classList.add('done');
    optimizeButton.disabled = false;
    if (optimizeLabel) optimizeLabel.textContent = '优化完成';
    if (memoryStatus) memoryStatus.textContent = '已应用安全方案';
  }, 900);
});
