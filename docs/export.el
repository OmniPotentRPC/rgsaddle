;;; Export every org page under docs/orgmode to reStructuredText under docs/source.
(require 'package)
(add-to-list 'package-archives '("gnu" . "https://elpa.gnu.org/packages/") t)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize)
(package-refresh-contents)
;; ox-rst calls org-element-type-p, which Org 9.7 added.
;; Emacs 29 ships Org 9.6.15, and package-installed-p counts that built-in,
;; so install the GNU ELPA package when the active Org is older than 9.7.
(unless (package-installed-p 'org '(9 7))
  (package-install (cadr (assq 'org package-archive-contents))))
(unless (package-installed-p 'ox-rst)
  (package-install 'ox-rst))
(require 'ox-rst)
(let* ((docs (file-name-directory (or load-file-name buffer-file-name)))
       (source (expand-file-name "orgmode" docs))
       (output (expand-file-name "source" docs)))
  (dolist (file (directory-files-recursively source "\\.org$"))
    (let ((relative (file-relative-name file source)))
      (with-current-buffer (find-file-noselect file)
        (let ((destination (expand-file-name (concat (file-name-sans-extension relative) ".rst") output)))
          (make-directory (file-name-directory destination) t)
          (org-export-to-file 'rst destination))))))
