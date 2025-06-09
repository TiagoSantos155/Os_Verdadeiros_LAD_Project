document.addEventListener('DOMContentLoaded', function () {
    // Aplica delays crescentes para animação em cada card
    document.querySelectorAll('.biblioteca-anim').forEach(function(card, idx) {
        card.style.setProperty('--anim-delay', (idx * 0.08) + 's');
    });
});
