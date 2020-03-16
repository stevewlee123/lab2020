package lab2020;
//import org.hibernate.loader.custom.Return;
import java.util.Iterator;
import java.awt.event.ItemEvent;


public class ResizingArrayStack<T> implements Iterable<T> {
    private T[] a = (T[]) new Object[1];
    private int N = 0;

    public boolean isEmpty() {
        return N == 0;
    }

    public int size() {
        return N;
    }

    private void resize(int max) {
        T[] temp = (T[]) new Object[max];
        for (int i = 0; i < N; i++) {
            temp[i] = a[i];
            a = temp;
        }
    }

    public void push(T item) {
        //判断当数组元素计数变量N与数组的长度相等的时候 那么将数组的长度扩充2倍
        if (N == a.length) {
            resize(2 * a.length);
        }
        a[N++] = item;
    }

//    public Item pop() {
//        Item item = a[N--];
//        a[N] = null;
//        if (N > 0 && N == a.length / 4) {
//            resize(a.length / 2);
//        }
//        return item;
//    }

    @Override
    public Iterator<T> iterator() {
        return new ReverseArrayIterator();
    }

    private class ReverseArrayIterator implements Iterator<T> {
        private int i = N;

        @Override
        public boolean hasNext() {
            return i > 0;
        }

        @Override
        public T next() {
            return a[--i];
        }

        @Override
        public void remove() {
        }

    }


}