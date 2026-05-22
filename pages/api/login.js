import { serialize } from 'cookie';

export default function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Método no permitido' });
  }

  const { password } = req.body;
  const expectedPassword = process.env.APP_PASSWORD;

  if (!expectedPassword) {
    return res.status(500).json({ message: 'Servidor mal configurado: APP_PASSWORD no definido.' });
  }

  if (password === expectedPassword) {
    // Set a secure, HTTP-only cookie containing the auth state
    const cookie = serialize('auth_token', 'authenticated', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7, // 1 week
      path: '/',
    });

    res.setHeader('Set-Cookie', cookie);
    return res.status(200).json({ success: true });
  }

  return res.status(401).json({ message: 'Contraseña incorrecta' });
}
