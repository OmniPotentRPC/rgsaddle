;;; Export every org page under docs/orgmode to reStructuredText under docs/source.
;;; ox-rst calls org-element-type-p, which Org 9.7 added. Ubuntu's Emacs 29
;;; ships Org 9.6, so install the current GNU ELPA Org release first.
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(setq package-install-upgrade-built-in t)
(package-initialize)
(defconst rgsaddle-min-org-version '(9 7))
(unless (package-installed-p 'org rgsaddle-min-org-version)
  (package-refresh-contents)
  (let ((desc (cadr (assq 'org package-archive-contents))))
    (unless (and desc
                 (version-list-<= rgsaddle-min-org-version
                                  (package-desc-version desc)))
      (error "No Org release >= 9.7 in the package archives"))
    (package-install desc)))
(unless (package-installed-p 'ox-rst)
  (unless package-archive-contents
    (package-refresh-contents))
  (package-install 'ox-rst))
(require 'ox-rst)
(unless (fboundp 'org-element-type-p)
  (error "org-element-type-p is missing from Org %s (%s)"
         (org-version)
         (locate-library "org")))
(message "Exporting with %s" (org-version nil t))
(let* ((docs (file-name-directory (or load-file-name buffer-file-name)))
       (source (expand-file-name "orgmode" docs))
       (output (expand-file-name "source" docs)))
  (dolist (file (directory-files-recursively source "\\.org$"))
    (let ((relative (file-relative-name file source)))
      (with-current-buffer (find-file-noselect file)
        (let ((destination (expand-file-name (concat (file-name-sans-extension relative) ".rst") output)))
          (make-directory (file-name-directory destination) t)
          (org-export-to-file 'rst destination))))))
