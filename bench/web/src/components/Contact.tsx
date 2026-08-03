/** Who to reach, on every page.
 *
 *  A benchmark asks people to hand over their data and then tells them
 *  something they may not like about it. That is a lot to ask of a site with no
 *  name attached to it, so the person behind it is reachable from anywhere in
 *  the product rather than from a page you have to go looking for.
 *
 *  `rel="me"` on the profile links is the small honest version of the same
 *  point: it is the standard way of asserting that these accounts and this site
 *  are the same person, and it is what verification tools read.
 *
 *  `noopener` on every external link, which is not optional. Without it the page
 *  being opened gets a handle on this one through `window.opener` and can
 *  navigate it somewhere else, and the visitor sees a tab they trusted quietly
 *  become a different site.
 */

const LINKS = [
  { label: "LinkedIn", href: "https://www.linkedin.com/in/gurasees-singh-dhanoa-9aa7a221b/" },
  { label: "X", href: "https://x.com/Gurasees0" },
  { label: "Email", href: "mailto:ae23b064@smail.iitm.ac.in", text: "ae23b064@smail.iitm.ac.in" },
];

export function Contact() {
  return (
    <footer className="contact">
      <div className="contact-inner">
        <div>
          <h2>Get in touch</h2>
          <p>
            Questions about a result, a dataset that will not upload, or anything the
            report got wrong. Built by Gurasees Singh Dhanoa.
          </p>
        </div>
        <ul className="contact-links">
          {LINKS.map((link) => (
            <li key={link.label}>
              <a
                href={link.href}
                rel={link.href.startsWith("mailto:") ? undefined : "me noopener noreferrer"}
                target={link.href.startsWith("mailto:") ? undefined : "_blank"}
              >
                {link.text ?? link.label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </footer>
  );
}
