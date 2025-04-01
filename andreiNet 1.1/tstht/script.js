document.querySelector('.container').addEventListener('mousemove', (e) => {
    const rect = e.target.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * 100;
    const y = (e.clientY - rect.top) / rect.height * 100;
    
    e.target.style.setProperty('--mouse-x', `${x}%`);
    e.target.style.setProperty('--mouse-y', `${y}%`);
});