(function () {
  var bookmarkPromptKey = 'bookmarkPromptShown';
  if (localStorage.getItem(bookmarkPromptKey)) {
    return;
  }

  var isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  var shortcut = isMac ? 'Command (⌘) + D' : 'Ctrl + D';
  var shouldShowTip = confirm('Enjoying WebberTech? Click OK to see how to bookmark this site to your browser bar.');

  if (shouldShowTip) {
    alert('To bookmark this website to your browser bar, press ' + shortcut + ' now.');
  }

  localStorage.setItem(bookmarkPromptKey, 'true');
})();
