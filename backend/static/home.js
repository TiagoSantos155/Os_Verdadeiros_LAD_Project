document.addEventListener('DOMContentLoaded', function() {
    // Banner carousel logic
    let top10GamesData = [];
    try {
        const top10Raw = document.getElementById('top10-data');
        if (top10Raw) {
            top10GamesData = JSON.parse(top10Raw.textContent)
                .filter(g => g.img)
                .map(g => ({img: g.img, title: g.name}));
        }
    } catch (e) {
        top10GamesData = [];
    }

    let bannerIdx = 0;
    function rotateBanner() {
        if (top10GamesData.length === 0) return;
        bannerIdx = (bannerIdx + 1) % top10GamesData.length;
        const imgElem = document.getElementById('banner-carousel-img');
        imgElem.src = top10GamesData[bannerIdx].img;
        imgElem.alt = top10GamesData[bannerIdx].title;
        imgElem.classList.add('fade-in');
        setTimeout(() => imgElem.classList.remove('fade-in'), 600);
    }
    if (top10GamesData.length > 1) {
        setInterval(rotateBanner, 5000);
    }

    // Modal de avaliação
    let selectedRating = 0;
    let selectedGameId = null;
    let selectedGameTitle = "";

    function setStars(val) {
        selectedRating = val;
        document.querySelectorAll('#star-container .star').forEach((star, idx) => {
            if (idx < val) {
                star.classList.add('selected');
            } else {
                star.classList.remove('selected');
            }
        });
    }

    document.querySelectorAll('.game-rate-trigger').forEach(card => {
        card.addEventListener('click', function() {
            selectedGameId = this.getAttribute('data-gameid');
            selectedGameTitle = this.getAttribute('data-title');
            document.getElementById('modal-game-title').textContent = selectedGameTitle;
            document.getElementById('rate-modal').classList.add('active');
            selectedRating = 0;
            setStars(0);
        });
    });

    document.getElementById('close-modal').onclick = function() {
        document.getElementById('rate-modal').classList.remove('active');
    };

    document.querySelectorAll('#star-container .star').forEach(star => {
        star.addEventListener('mouseenter', function() {
            let val = parseInt(this.getAttribute('data-value'));
            document.querySelectorAll('#star-container .star').forEach((s, idx) => {
                if (idx < val) {
                    s.classList.add('active');
                } else {
                    s.classList.remove('active');
                }
            });
        });
        star.addEventListener('mouseleave', function() {
            document.querySelectorAll('#star-container .star').forEach((s, idx) => {
                s.classList.remove('active');
            });
            setStars(selectedRating);
        });
        star.addEventListener('click', function() {
            let val = parseInt(this.getAttribute('data-value'));
            setStars(val);
        });
    });

    document.getElementById('submit-rating').onclick = function() {
        if (!selectedRating || !selectedGameId) return;
        fetch('/rate_game', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({gameid: selectedGameId, rating: selectedRating})
        }).then(res => res.json()).then(data => {
            document.getElementById('rate-modal').classList.remove('active');
        });
    };

    // Biblioteca do utilizador
    let userLibraryData = [];
    try {
        const libRaw = document.getElementById('user-library-data');
        if (libRaw) {
            userLibraryData = JSON.parse(libRaw.textContent);
        }
    } catch (e) {
        userLibraryData = [];
    }

    // Adicionar à biblioteca
    function setupAddToLibraryButtons() {
        document.querySelectorAll('.add-library-btn').forEach(btn => {
            const gameid = parseInt(btn.getAttribute('data-gameid'));
            // Desativa botão se já estiver na biblioteca (extra segurança JS)
            if (userLibraryData.includes(gameid)) {
                btn.disabled = true;
                btn.textContent = "Adicionado!";
                btn.style.background = "#66c0f4";
                btn.style.color = "#fff";
                return;
            }
            if (!btn.dataset.listener) {
                btn.addEventListener('click', function() {
                    fetch('/add_to_library', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({gameid: gameid})
                    }).then(res => res.json()).then(data => {
                        if (data.success) {
                            btn.textContent = "Adicionado!";
                            btn.disabled = true;
                            btn.style.background = "#66c0f4";
                            btn.style.color = "#fff";
                            // Atualiza array local para impedir novo clique sem refresh
                            userLibraryData.push(gameid);
                        }
                    });
                });
                btn.dataset.listener = "true";
            }
        });
    }
    setupAddToLibraryButtons();

    // Fade-in animation for banner
    const style = document.createElement('style');
    style.innerHTML = `
    .fade-in {
        animation: fadeInBanner 0.6s;
    }
    @keyframes fadeInBanner {
        from { opacity: 0.5; filter: blur(2px);}
        to { opacity: 1; filter: none;}
    }`;
    document.head.appendChild(style);

    // Banner shrink on scroll
    const banner = document.querySelector('.steam-banner');
    function handleBannerResize() {
        if (!banner) return;
        if (window.scrollY > 80) {
            banner.classList.add('banner-small');
        } else {
            banner.classList.remove('banner-small');
        }
    }
    window.addEventListener('scroll', handleBannerResize);

    document.querySelectorAll('.horizontal-scroll-wrapper').forEach(function(wrapper) {
        const row = wrapper.querySelector('.row.overflow-auto');
        const leftBtn = wrapper.querySelector('.scroll-arrow.left');
        const rightBtn = wrapper.querySelector('.scroll-arrow.right');
        if (!row || !leftBtn || !rightBtn) return;

        leftBtn.addEventListener('click', function() {
            row.scrollBy({ left: -300, behavior: 'smooth' });
        });
        rightBtn.addEventListener('click', function() {
            row.scrollBy({ left: 300, behavior: 'smooth' });
        });
    });
});
