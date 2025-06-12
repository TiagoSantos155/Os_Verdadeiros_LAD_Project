document.addEventListener('DOMContentLoaded', function() {
    // Highlight selected genres
    const genreLabels = document.querySelectorAll('.genre-checkbox-label');
    genreLabels.forEach(label => {
        const checkbox = label.querySelector('input[type="checkbox"]');
        if (checkbox.checked) label.classList.add('selected');
        label.addEventListener('click', function(e) {
            if (e.target.tagName !== 'INPUT') {
                checkbox.checked = !checkbox.checked;
            }
            label.classList.toggle('selected', checkbox.checked);
            updateGenreCount();
            updateSelectedGenres();
        });
        checkbox.addEventListener('change', function() {
            label.classList.toggle('selected', checkbox.checked);
            updateGenreCount();
            updateSelectedGenres();
        });
    });

    // Add genre count display
    const genreGroup = document.querySelector('.genre-checkbox-group');
    let counter, selectedList;
    if (genreGroup) {
        counter = document.createElement('div');
        counter.className = 'genre-counter';
        counter.style.marginTop = '8px';
        counter.style.fontSize = '0.95em';
        counter.style.color = '#6ad1ff';
        genreGroup.parentNode.appendChild(counter);

        // Add selected genres list
        selectedList = document.createElement('div');
        selectedList.className = 'selected-genres-list';
        selectedList.style.marginTop = '4px';
        selectedList.style.fontSize = '0.97em';
        selectedList.style.color = '#fff';
        genreGroup.parentNode.appendChild(selectedList);

        function updateGenreCount() {
            const checked = genreGroup.querySelectorAll('input[type="checkbox"]:checked').length;
            counter.textContent = checked === 0
                ? 'Nenhum género selecionado'
                : `${checked} género${checked > 1 ? 's' : ''} selecionado${checked > 1 ? 's' : ''}`;
        }
        function updateSelectedGenres() {
            const checked = genreGroup.querySelectorAll('input[type="checkbox"]:checked');
            if (checked.length === 0) {
                selectedList.textContent = '';
                return;
            }
            const names = Array.from(checked).map(cb => cb.parentNode.textContent.trim());
            selectedList.innerHTML = `<span style="color:#6ad1ff;">Selecionados:</span> ${names.join(', ')}`;
        }
        window.updateGenreCount = updateGenreCount;
        window.updateSelectedGenres = updateSelectedGenres;
        updateGenreCount();
        updateSelectedGenres();
    }

    // Dynamic genre select logic
    const genresList = window.GENRES_LIST || [];
    const input = document.getElementById('genre-search');
    const suggestionsBox = document.getElementById('genre-suggestions');
    const selectedBox = document.getElementById('selected-genres');
    const hiddenInput = document.getElementById('genres-hidden');
    let selectedGenres = [];

    function renderSuggestions(filter) {
        suggestionsBox.innerHTML = '';
        if (!filter) {
            suggestionsBox.style.display = 'none';
            return;
        }
        const filterLower = filter.toLowerCase();
        const filtered = genresList.filter(
            g => g.toLowerCase().includes(filterLower) && !selectedGenres.includes(g)
        );
        if (filtered.length === 0) {
            suggestionsBox.style.display = 'none';
            return;
        }
        filtered.forEach((genre, idx) => {
            const div = document.createElement('div');
            div.className = 'genre-suggestion-item';
            div.textContent = genre;
            div.addEventListener('mousedown', function(e) {
                e.preventDefault();
                addGenre(genre);
                input.value = '';
                suggestionsBox.style.display = 'none';
            });
            suggestionsBox.appendChild(div);
        });
        suggestionsBox.style.display = 'block';
    }

    function renderSelected() {
        selectedBox.innerHTML = '';
        selectedGenres.forEach(genre => {
            const chip = document.createElement('span');
            chip.className = 'genre-chip';
            chip.textContent = genre;
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-genre';
            removeBtn.type = 'button';
            removeBtn.innerHTML = '&times;';
            removeBtn.title = 'Remover';
            removeBtn.onclick = () => {
                selectedGenres = selectedGenres.filter(g => g !== genre);
                renderSelected();
                updateHidden();
            };
            chip.appendChild(removeBtn);
            selectedBox.appendChild(chip);
        });
        updateHidden();
    }

    function addGenre(genre) {
        if (!selectedGenres.includes(genre)) {
            selectedGenres.push(genre);
            renderSelected();
        }
    }

    function updateHidden() {
        hiddenInput.value = selectedGenres.join(',');
    }

    // Allow enter to add custom genre
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && input.value.trim()) {
            e.preventDefault();
            const val = input.value.trim();
            if (!selectedGenres.includes(val)) {
                addGenre(val);
            }
            input.value = '';
            suggestionsBox.style.display = 'none';
        } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            // Keyboard navigation for suggestions
            const items = suggestionsBox.querySelectorAll('.genre-suggestion-item');
            if (items.length === 0) return;
            let idx = Array.from(items).findIndex(item => item.classList.contains('active'));
            if (e.key === 'ArrowDown') {
                idx = (idx + 1) % items.length;
            } else {
                idx = (idx - 1 + items.length) % items.length;
            }
            items.forEach(item => item.classList.remove('active'));
            items[idx].classList.add('active');
            items[idx].scrollIntoView({block: 'nearest'});
            e.preventDefault();
        } else if (e.key === 'Tab') {
            suggestionsBox.style.display = 'none';
        }
    });

    input.addEventListener('input', function() {
        renderSuggestions(input.value.trim());
    });

    input.addEventListener('focus', function() {
        if (input.value.trim()) renderSuggestions(input.value.trim());
    });

    input.addEventListener('blur', function() {
        setTimeout(() => suggestionsBox.style.display = 'none', 120);
    });

    suggestionsBox.addEventListener('mousedown', function(e) {
        e.preventDefault();
    });

    // Click on suggestion with mouse
    suggestionsBox.addEventListener('mouseover', function(e) {
        if (e.target.classList.contains('genre-suggestion-item')) {
            Array.from(suggestionsBox.children).forEach(child => child.classList.remove('active'));
            e.target.classList.add('active');
        }
    });

    suggestionsBox.addEventListener('click', function(e) {
        if (e.target.classList.contains('genre-suggestion-item')) {
            addGenre(e.target.textContent);
            input.value = '';
            suggestionsBox.style.display = 'none';
        }
    });

    // Remove genre with backspace
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Backspace' && !input.value && selectedGenres.length > 0) {
            selectedGenres.pop();
            renderSelected();
        }
    });

    // On form submit, ensure hidden input is updated
    const form = document.querySelector('.dev-add-game-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            updateHidden();
            const btn = form.querySelector('button[type="submit"]');
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'A processar...';
                btn.style.opacity = '0.7';
            }
        });
    }

    // Preencher géneros selecionados vindos do backend
    if (window.SELECTED_GENRES_FROM_SERVER) {
        let genres = window.SELECTED_GENRES_FROM_SERVER.split(',').map(g => g.trim()).filter(g => g);
        genres.forEach(g => addGenre(g));
        input.value = '';
        suggestionsBox.style.display = 'none';
    }

    // Initial render
    renderSelected();
});
